#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profile SKLT sparse attention backend using vLLM's profiler.

This script profiles the SKLT sparse attention backend with streaming indexer
using vLLM's built-in torch profiler, similar to profile_vllm.py.

It generates perfetto/chrome traces that can be viewed in chrome://tracing

Example usage:
    # Profile SKLT with default settings
    python profile_sklt_vllm.py
    
    # Profile with custom sparse configuration
    python profile_sklt_vllm.py --sink-tokens 4 --window-size 512
    
    # Profile with longer context
    python profile_sklt_vllm.py --context-length 32768

The profiler traces will be saved to: ./vllm_profile_sklt/

Requirements:
    - vLLM with SKLT backend installed
    - CUDA-capable GPU
    - PyTorch with CUDA support
"""

import argparse
import sys
import time

try:
    from vllm import LLM, SamplingParams
    from vllm.config.attention import AttentionConfig, IndexerConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
except ImportError as e:
    print(f"Error: vLLM is required. Install it with: pip install -e .")
    print(f"Import error details: {e}")
    sys.exit(1)


def get_prompts(token_length, tokenizer):
    """Create a prompt with specified token length."""
    if token_length <= 0:
        return ""
    
    base_text = (
        "Hello, my name is John Doe. I am a software engineer. "
        "I live in New York City. "
    )
    
    def encode(text):
        if hasattr(tokenizer, "encode"):
            try:
                return tokenizer.encode(text, add_special_tokens=False)
            except TypeError:
                return tokenizer.encode(text)
        encoded = tokenizer(text, add_special_tokens=False)
        return getattr(encoded, "input_ids", encoded)
    
    def decode(token_ids):
        if hasattr(tokenizer, "decode"):
            try:
                return tokenizer.decode(token_ids, skip_special_tokens=True)
            except TypeError:
                return tokenizer.decode(token_ids)
        return tokenizer.decode(token_ids)
    
    base_tokens = encode(base_text)
    if not base_tokens:
        return base_text
    
    tokens = []
    while len(tokens) < token_length:
        tokens.extend(base_tokens)
    
    tokens = tokens[:token_length]
    return decode(tokens)


def main():
    parser = argparse.ArgumentParser(
        description="Profile SKLT sparse attention backend using vLLM profiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--context-length",
        type=int,
        default=128000,
        help="Context length (number of tokens in prompt)",
    )
    
    parser.add_argument(
        "--sink-tokens",
        type=int,
        default=4,
        help="Number of sink tokens (always attended)",
    )
    
    parser.add_argument(
        "--window-size",
        type=int,
        default=512,
        help="Local attention window size",
    )
    
    parser.add_argument(
        "--max-sparse-k",
        type=int,
        default=1024,
        help="Maximum sparse keys per head (buffer size)",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Model to profile",
    )
    
    parser.add_argument(
        "--profiler-dir",
        type=str,
        default="./vllm_profile_sklt",
        help="Directory to save profiler traces",
    )
    
    parser.add_argument(
        "--num-decode-steps",
        type=int,
        default=4,
        help="Number of decode steps to profile (max_tokens)",
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"SKLT Sparse Attention Profiling with vLLM Profiler")
    print(f"{'='*80}")
    print(f"  Model: {args.model}")
    print(f"  Context length: {args.context_length}")
    print(f"  Decode steps: {args.num_decode_steps}")
    print(f"  Sink tokens: {args.sink_tokens}")
    print(f"  Window size: {args.window_size}")
    print(f"  Max sparse k: {args.max_sparse_k}")
    print(f"  Profiler output: {args.profiler_dir}")
    print(f"{'='*80}\n")
    
    # Configure SKLT backend
    indexer_config = IndexerConfig(
        indexer_type="streaming",
        num_sink_tokens=args.sink_tokens,
        local_window_size=args.window_size,
        max_sparse_k=args.max_sparse_k,
    )
    
    attention_config = AttentionConfig(
        backend=AttentionBackendEnum.SKLT,
        indexer_config=indexer_config,
        use_sklt_sparse_attention=True,
    )
    
    print("Initializing LLM with SKLT backend and profiler...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        attention_config=attention_config,
        profiler_config={
            "profiler": "torch",
            "torch_profiler_dir": args.profiler_dir,
        },
        max_model_len=args.context_length + 20,
        enforce_eager=True,  # Required for SKLT PyTorch implementation (no CUDA graphs)
    )
    
    print("✓ LLM initialized")
    
    # Create prompt with specified context length
    print(f"\nCreating prompt with {args.context_length} tokens...")
    prompt = get_prompts(args.context_length, llm.get_tokenizer())
    print(f"  Prompt length: {len(prompt)} chars")
    
    # Create sampling params
    sampling_params = SamplingParams(
        max_tokens=args.num_decode_steps,
        temperature=0.8,
        top_p=0.95,
    )
    
    # Warmup (2 runs)
    print("\nWarming up (2 iterations)...")
    for _ in range(2):
        llm.generate([prompt], sampling_params)
    
    print("✓ Warmup complete")
    
    # Profile generation
    print(f"\nStarting profiling (will generate {args.num_decode_steps} tokens)...")
    print("  Profiler is active - this may be slower than normal")
    
    #llm.start_profile()
    outputs = llm.generate([prompt], sampling_params)
    #llm.stop_profile()
    
    print("✓ Profiling complete")
    
    # Print results
    print(f"\n{'='*80}")
    print("Generation Results")
    print(f"{'='*80}")
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Prompt length: {len(output.prompt)} chars")
        print(f"Generated: {generated_text[:100]}{'...' if len(generated_text) > 100 else ''}")
        print(f"Tokens generated: {len(output.outputs[0].token_ids)}")
    
    # Wait for profiler to finish writing
    print(f"\nWaiting for profiler to finish writing traces...")
    time.sleep(10)
    
    print(f"\n{'='*80}")
    print("Profiling Complete!")
    print(f"{'='*80}")
    print(f"Traces saved to: {args.profiler_dir}")
    print(f"\nTo view traces:")
    print(f"  1. Open chrome://tracing in Chrome browser")
    print(f"  2. Load trace file from {args.profiler_dir}")
    print(f"  3. Look for 'sklt_sparse_attention' operations")
    print(f"\nSparse pattern used:")
    print(f"  - Sink tokens: {args.sink_tokens}")
    print(f"  - Window size: {args.window_size}")
    print(f"  - Total sparse keys: ~{args.sink_tokens + args.window_size}")
    print(f"  - Full context: {args.context_length}")
    print(f"  - Compute reduction: ~{(1 - (args.sink_tokens + args.window_size)/args.context_length)*100:.1f}%")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
