#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profile attention backend forward pass for DECODE (token generation).

This script profiles the DECODE scenario: generating one new token when the
KV cache already contains 'context_length' tokens. This measures the latency
and throughput for token generation during inference.

Key setup:
    - Query length = 1 (generating 1 new token)
    - KV cache size = context_length (already computed tokens)
    - Metadata seq_lens = context_length (total sequence length including new token)

Example usage:
    # Profile FlashAttention with 4096 tokens in KV cache
    python profile_attention.py --backend FLASH_ATTN --context-length 4096
    
    # Profile multiple context lengths
    python profile_attention.py --backend FLASH_ATTN --context-length 1024 2048 4096 8192
    
    # Profile with different batch sizes
    python profile_attention.py --backend FLASH_ATTN --context-length 4096 --batch-size 1 4 8 16
    
    # Use torch profiler for detailed analysis
    python profile_attention.py --backend FLASH_ATTN --context-length 4096 --use-profiler

Requirements:
    - vLLM must be installed: pip install -e .
    - CUDA-capable GPU
    - PyTorch with CUDA support
"""

import argparse
import sys
import time
from dataclasses import dataclass
from typing import List

try:
    import torch
    import torch.nn as nn
except ImportError as e:
    print(f"Error: PyTorch is required. Install it with: pip install torch")
    sys.exit(1)

try:
    from vllm.config import (
        CacheConfig,
        CompilationConfig,
        DeviceConfig,
        LoadConfig,
        ModelConfig,
        ParallelConfig,
        SchedulerConfig,
        VllmConfig,
    )
    from vllm.config.model import ModelDType
    from vllm.v1.attention.backend import AttentionType, CommonAttentionMetadata
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.kv_cache_interface import FullAttentionSpec
except ImportError as e:
    print(f"Error: vLLM is required. Install it with: pip install -e .")
    print(f"Import error details: {e}")
    sys.exit(1)


@dataclass
class ProfileConfig:
    """Configuration for profiling."""
    backend: str
    context_lengths: List[int]
    batch_sizes: List[int]
    num_warmup: int
    num_iterations: int
    use_profiler: bool
    use_cuda_graph: bool
    dtype: str
    num_heads: int
    num_kv_heads: int
    head_size: int
    block_size: int


class MockQKVLinear(nn.Module):
    """Mock QKV linear layer for testing."""
    
    def __init__(self, hidden_size: int, num_heads: int, head_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = head_size
        # Mock scale tensors for FP8 quantization support
        self._q_scale = torch.tensor([1.0], device="cuda")
        self._k_scale = torch.tensor([1.0], device="cuda")
        self._v_scale = torch.tensor([1.0], device="cuda")


def create_vllm_config(
    dtype: str = "float16",
    num_heads: int = 32,
    num_kv_heads: int = 32,
    head_size: int = 128,
    block_size: int = 16,
    max_model_len: int = 8192,
) -> VllmConfig:
    """Create a VllmConfig for profiling."""
    
    model_config = ModelConfig(
        model="Qwen/Qwen3-4B-Instruct-2507",
        tokenizer="Qwen/Qwen3-4B-Instruct-2507",
        trust_remote_code=False,
        dtype=dtype,
        seed=0,
        max_model_len=max_model_len,
    )
    
    cache_config = CacheConfig(
        block_size=block_size,
        cache_dtype="auto",
        swap_space=0,
    )
    cache_config.num_gpu_blocks = 10000
    cache_config.num_cpu_blocks = 0
    
    parallel_config = ParallelConfig(tensor_parallel_size=1)
    
    scheduler_config = SchedulerConfig(
        max_num_seqs=256,
        max_num_batched_tokens=8192,
        enable_chunked_prefill=True,
        max_model_len=model_config.max_model_len,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    
    device_config = DeviceConfig()
    load_config = LoadConfig()
    compilation_config = CompilationConfig()
    
    # Add mock methods
    import types
    model_config.get_num_layers = types.MethodType(lambda self: 1, model_config)
    model_config.get_sliding_window_for_layer = types.MethodType(
        lambda self, i: None, model_config
    )
    model_config.get_logits_soft_cap_for_layer = types.MethodType(
        lambda self, i: 0.0, model_config
    )
    model_config.get_sm_scale_for_layer = types.MethodType(
        lambda self, i: 1.0 / head_size ** 0.5, model_config
    )
    
    return VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        device_config=device_config,
        load_config=load_config,
        compilation_config=compilation_config,
    )


def create_common_attn_metadata(
    batch_size: int,
    context_length: int,
    query_length: int,
    block_size: int,
    device: torch.device,
) -> CommonAttentionMetadata:
    """Create CommonAttentionMetadata for profiling."""
    
    # All sequences have the same length for simplicity
    seq_lens = [context_length] * batch_size
    query_lens = [query_length] * batch_size
    
    # Create query start locations
    query_start_loc = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    query_start_loc[1:] = torch.tensor(query_lens, dtype=torch.int32, device=device).cumsum(0)
    query_start_loc_cpu = query_start_loc.cpu()
    num_tokens = sum(query_lens)
    
    # Create sequence lengths
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    seq_lens_cpu = seq_lens_tensor.cpu()
    max_seq_len = context_length
    
    # Create context lengths
    context_lens = [seq_lens[i] - query_lens[i] for i in range(batch_size)]
    num_computed_tokens_cpu = torch.tensor(context_lens, dtype=torch.int32)
    
    # Create block table and slot mapping
    max_blocks = (context_length + block_size - 1) // block_size
    # Use sequential block indices for simplicity
    block_table_tensor = torch.arange(
        batch_size * max_blocks, dtype=torch.int32, device=device
    ).view(batch_size, max_blocks)
    
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    
    max_query_len = query_length
    
    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens_tensor,
        _seq_lens_cpu=seq_lens_cpu,
        _num_computed_tokens_cpu=num_computed_tokens_cpu,
        num_reqs=batch_size,
        num_actual_tokens=num_tokens,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        causal=True,
    )


def create_kv_cache(
    backend_class,
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Create a KV cache tensor with random data."""
    
    kv_cache_shape = backend_class.get_kv_cache_shape(
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
    )
    
    kv_cache = torch.randn(kv_cache_shape, dtype=dtype, device=device)
    return kv_cache


def profile_attention_forward(
    config: ProfileConfig,
    context_length: int,
    batch_size: int,
):
    """Profile the forward pass of an attention backend."""
    
    print(f"\n{'='*80}")
    print(f"Profiling {config.backend} with context_length={context_length}, batch_size={batch_size}")
    print(f"{'='*80}")
    
    device = torch.device("cuda")
    dtype = torch.float16 if config.dtype == "float16" else torch.bfloat16
    
    # Get backend
    try:
        backend_enum = AttentionBackendEnum[config.backend]
        backend_class = backend_enum.get_class()
    except (KeyError, ImportError) as e:
        print(f"Error: Backend {config.backend} not available: {e}")
        return
    
    # Create VllmConfig
    vllm_config = create_vllm_config(
        dtype=config.dtype,
        num_heads=config.num_heads,
        num_kv_heads=config.num_kv_heads,
        head_size=config.head_size,
        block_size=config.block_size,
        max_model_len=max(config.context_lengths) + 1024,
    )
    
    # Create attention spec
    attn_spec = FullAttentionSpec(
        block_size=config.block_size,
        num_kv_heads=config.num_kv_heads,
        head_size=config.head_size,
        dtype=dtype,
        sliding_window=None,
    )
    
    # Create attention implementation
    impl_cls = backend_class.get_impl_cls()
    attn_impl = impl_cls(
        num_heads=config.num_heads,
        head_size=config.head_size,
        scale=1.0 / (config.head_size ** 0.5),
        num_kv_heads=config.num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type=AttentionType.DECODER,
    )
    
    # Create metadata builder
    builder_cls = backend_class.get_builder_cls()
    metadata_builder = builder_cls(
        kv_cache_spec=attn_spec,
        layer_names=["attention"],  # Mock layer name
        vllm_config=vllm_config,
        device=device,
    )
    
    # DECODE SCENARIO: query_length = 1 (generating one new token)
    # The KV cache already contains 'context_length' tokens
    # This measures the time to generate one new token given a context
    query_length = 1
    
    # Create common attention metadata
    # seq_lens = context_length means the KV cache has context_length tokens
    # query_lens = 1 means we're generating 1 new token
    common_metadata = create_common_attn_metadata(
        batch_size=batch_size,
        context_length=context_length,
        query_length=query_length,
        block_size=config.block_size,
        device=device,
    )
    
    # Build backend-specific metadata
    # common_prefix_len=0 means no prefix sharing between sequences
    attn_metadata = metadata_builder.build(
        common_prefix_len=0,
        common_attn_metadata=common_metadata,
    )
    
    # Create KV cache
    num_blocks = (context_length * batch_size + config.block_size - 1) // config.block_size + 100
    kv_cache = create_kv_cache(
        backend_class=backend_class,
        num_blocks=num_blocks,
        block_size=config.block_size,
        num_kv_heads=config.num_kv_heads,
        head_size=config.head_size,
        dtype=dtype,
        device=device,
    )
    
    # Create input tensors
    num_tokens = batch_size * query_length
    hidden_size = config.num_heads * config.head_size
    
    query = torch.randn(num_tokens, config.num_heads, config.head_size, dtype=dtype, device=device)
    key = torch.randn(num_tokens, config.num_kv_heads, config.head_size, dtype=dtype, device=device)
    value = torch.randn(num_tokens, config.num_kv_heads, config.head_size, dtype=dtype, device=device)
    
    # Create output tensor - must match query shape for attention
    output = torch.empty(num_tokens, config.num_heads, config.head_size, dtype=dtype, device=device)
    
    # Create mock layer
    mock_layer = MockQKVLinear(hidden_size, config.num_heads, config.head_size)
    
    print(f"\nModel Configuration:")
    print(f"  - Num heads: {config.num_heads}")
    print(f"  - Num KV heads: {config.num_kv_heads}")
    print(f"  - Head size: {config.head_size}")
    print(f"  - Hidden size: {hidden_size}")
    print(f"  - Block size: {config.block_size}")
    print(f"  - Dtype: {dtype}")
    
    print(f"\nSetup complete (DECODE mode - measuring time to generate 1 new token):")
    print(f"\nInput Tensor Shapes:")
    print(f"  - Query shape: {query.shape} (batch_size={batch_size}, query_len=1)")
    print(f"  - Key shape: {key.shape} (for new token)")
    print(f"  - Value shape: {value.shape} (for new token)")
    print(f"  - Output shape: {output.shape}")
    print(f"\nKV Cache:")
    print(f"  - KV cache shape: {kv_cache.shape}")
    print(f"  - KV cache dtype: {kv_cache.dtype}")
    print(f"  - KV cache device: {kv_cache.device}")
    print(f"  - KV cache contains: {context_length} tokens per sequence")
    print(f"  - Num blocks: {num_blocks}")
    print(f"\nMetadata:")
    print(f"  - Num query tokens: {num_tokens}")
    print(f"  - Num reqs: {common_metadata.num_reqs}")
    print(f"  - Num actual tokens: {common_metadata.num_actual_tokens}")
    print(f"  - Seq lens shape: {common_metadata.seq_lens.shape}")
    print(f"  - Seq lens (sample): {common_metadata.seq_lens.tolist()[:min(3, batch_size)]}{'...' if batch_size > 3 else ''}")
    print(f"  - Max seq len: {common_metadata.max_seq_len}")
    print(f"  - Max query len: {common_metadata.max_query_len}")
    print(f"  - Query start loc shape: {common_metadata.query_start_loc.shape}")
    print(f"  - Block table shape: {common_metadata.block_table_tensor.shape}")
    print(f"  - Slot mapping shape: {common_metadata.slot_mapping.shape}")
    
    # Log backend-specific metadata shapes if available
    if hasattr(attn_metadata, '__dict__'):
        print(f"\nBackend-Specific Metadata Attributes:")
        for attr_name, attr_value in attn_metadata.__dict__.items():
            if isinstance(attr_value, torch.Tensor):
                print(f"  - {attr_name} shape: {attr_value.shape}, dtype: {attr_value.dtype}")
    
    # Warmup
    print(f"\nWarming up ({config.num_warmup} iterations)...")
    try:
        for _ in range(config.num_warmup):
            attn_impl.forward(
                layer=mock_layer,
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                output=output,
            )
        torch.cuda.synchronize()
    except (RuntimeError, torch.cuda.CudaError) as e:
        error_msg = str(e)
        if "PTX" in error_msg or "UnsupportedPtxVersion" in error_msg:
            print(f"\n{'='*80}")
            print(f"ERROR: CUDA PTX Compatibility Issue")
            print(f"{'='*80}")
            print(f"Backend {config.backend} is not compatible with your GPU/CUDA setup.")
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"Compute Capability: {torch.cuda.get_device_capability()}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"\nThis typically happens when:")
            print(f"  1. FlashAttention was compiled for an older GPU architecture")
            print(f"  2. Your GPU (Blackwell/B200) requires newer compilation")
            print(f"\nSuggested solutions:")
            print(f"  1. Try a different backend: --backend TRITON_ATTN")
            print(f"  2. Recompile FlashAttention for your GPU architecture")
            print(f"  3. Use FlashAttention 3 if available")
            print(f"\nOriginal error: {error_msg}")
            print(f"{'='*80}\n")
            return None
        else:
            # Re-raise other errors
            raise
    
    # Benchmark
    print(f"Benchmarking ({config.num_iterations} iterations)...")
    
    # CUDA Graph support
    cuda_graph = None
    if config.use_cuda_graph:
        print(f"\nRecording CUDA graph...")
        # Create a CUDA graph
        cuda_graph = torch.cuda.CUDAGraph()
        
        # Warmup before recording (3 extra iterations for graph recording)
        for _ in range(3):
            attn_impl.forward(
                layer=mock_layer,
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                output=output,
            )
        torch.cuda.synchronize()
        
        # Record the graph
        with torch.cuda.graph(cuda_graph):
            attn_impl.forward(
                layer=mock_layer,
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                output=output,
            )
        torch.cuda.synchronize()
        print(f"CUDA graph recorded successfully!")
    
    if config.use_profiler:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
        ) as prof:
            for _ in range(config.num_iterations):
                if cuda_graph is not None:
                    cuda_graph.replay()
                else:
                    attn_impl.forward(
                        layer=mock_layer,
                        query=query,
                        key=key,
                        value=value,
                        kv_cache=kv_cache,
                        attn_metadata=attn_metadata,
                        output=output,
                    )
            torch.cuda.synchronize()
        
        # Print profiler results
        print("\nProfiler Results (Top 20 by CUDA time):")
        print(prof.key_averages().table(
            sort_by="cuda_time_total", 
            row_limit=20,
            max_name_column_width=80
        ))
        
        # Print profiler results with shapes
        print("\nProfiler Results with Input Shapes (Top 15 by CUDA time):")
        print(prof.key_averages(group_by_input_shape=True).table(
            sort_by="cuda_time_total",
            row_limit=15,
            max_name_column_width=60,
            max_shapes_column_width=80
        ))
        
        # Save trace
        trace_file = f"trace_{config.backend}_ctx{context_length}_bs{batch_size}.json"
        prof.export_chrome_trace(trace_file)
        print(f"\nTrace saved to: {trace_file}")
        print(f"View the trace at chrome://tracing")
    
    # Measure latency
    start_time = time.perf_counter()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(config.num_iterations):
        if cuda_graph is not None:
            cuda_graph.replay()
        else:
            attn_impl.forward(
                layer=mock_layer,
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                output=output,
            )
    end_event.record()
    torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    
    # Calculate metrics
    gpu_time_ms = start_event.elapsed_time(end_event)
    avg_latency_ms = gpu_time_ms / config.num_iterations
    cpu_time_ms = (end_time - start_time) * 1000
    avg_cpu_latency_ms = cpu_time_ms / config.num_iterations
    
    throughput = (batch_size * config.num_iterations) / (gpu_time_ms / 1000)  # sequences/sec
    
    # Calculate FLOPs (approximate)
    # Attention FLOPs ? 4 * batch_size * num_heads * query_len * context_len * head_size
    # (2 for QK^T matmul, 2 for softmax(QK^T)V matmul)
    flops_per_forward = 4 * batch_size * config.num_heads * query_length * context_length * config.head_size
    total_flops = flops_per_forward * config.num_iterations
    tflops_per_sec = (total_flops / (gpu_time_ms / 1000)) / 1e12
    
    print(f"\n{'='*80}")
    print(f"Results:")
    print(f"{'='*80}")
    print(f"Average GPU latency: {avg_latency_ms:.3f} ms")
    print(f"Average CPU latency: {avg_cpu_latency_ms:.3f} ms")
    print(f"Throughput: {throughput:.2f} sequences/sec")
    print(f"Estimated TFLOPS: {tflops_per_sec:.2f}")
    print(f"Memory bandwidth (approx): {(kv_cache.numel() * kv_cache.element_size() / (avg_latency_ms / 1000)) / 1e9:.2f} GB/s")
    print(f"{'='*80}\n")
    
    return {
        "backend": config.backend,
        "context_length": context_length,
        "batch_size": batch_size,
        "avg_latency_ms": avg_latency_ms,
        "throughput": throughput,
        "tflops": tflops_per_sec,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Profile attention backend forward pass",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--backend",
        type=str,
        default="FLASH_ATTN",
        choices=[e.name for e in AttentionBackendEnum],
        help="Attention backend to profile",
    )
    
    parser.add_argument(
        "--context-length",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096],
        help="Context lengths to profile (can specify multiple)",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        nargs="+",
        default=[1, 4, 8],
        help="Batch sizes to profile (can specify multiple)",
    )
    
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=10,
        help="Number of warmup iterations",
    )
    
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations",
    )
    
    parser.add_argument(
        "--use-profiler",
        action="store_true",
        help="Use torch profiler for detailed analysis",
    )
    
    parser.add_argument(
        "--use-cuda-graph",
        action="store_true",
        help="Use CUDA graph for benchmarking (record once, replay multiple times)",
    )
    
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16"],
        help="Data type for tensors",
    )
    
    parser.add_argument(
        "--num-heads",
        type=int,
        default=32,
        help="Number of attention heads",
    )
    
    parser.add_argument(
        "--num-kv-heads",
        type=int,
        default=8,
        help="Number of KV heads (for GQA)",
    )
    
    parser.add_argument(
        "--head-size",
        type=int,
        default=128,
        help="Size of each attention head",
    )
    
    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="KV cache block size",
    )
    
    args = parser.parse_args()
    
    config = ProfileConfig(
        backend=args.backend,
        context_lengths=args.context_length,
        batch_sizes=args.batch_size,
        num_warmup=args.num_warmup,
        num_iterations=args.num_iterations,
        use_profiler=args.use_profiler,
        use_cuda_graph=args.use_cuda_graph,
        dtype=args.dtype,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        head_size=args.head_size,
        block_size=args.block_size,
    )
    
    print(f"Profiling Configuration:")
    print(f"  Backend: {config.backend}")
    print(f"  Context lengths: {config.context_lengths}")
    print(f"  Batch sizes: {config.batch_sizes}")
    print(f"  Num warmup: {config.num_warmup}")
    print(f"  Num iterations: {config.num_iterations}")
    print(f"  Use profiler: {config.use_profiler}")
    print(f"  Use CUDA graph: {config.use_cuda_graph}")
    print(f"  Dtype: {config.dtype}")
    print(f"  Num heads: {config.num_heads}")
    print(f"  Num KV heads: {config.num_kv_heads}")
    print(f"  Head size: {config.head_size}")
    print(f"  Block size: {config.block_size}")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("\nError: CUDA is not available. This script requires a GPU.")
        return
    
    print(f"\nGPU: {torch.cuda.get_device_name()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Check for known compatibility issues
    gpu_name = torch.cuda.get_device_name()
    compute_cap = torch.cuda.get_device_capability()
    if "B200" in gpu_name or compute_cap[0] >= 10:
        if config.backend == "FLASH_ATTN":
            print(f"\nWARNING: {gpu_name} (compute capability {compute_cap[0]}.{compute_cap[1]}) may have compatibility issues with FLASH_ATTN.")
            print(f"If you encounter PTX errors, try: --backend TRITON_ATTN")
            print()
    
    # Profile all combinations
    all_results = []
    for context_length in config.context_lengths:
        for batch_size in config.batch_sizes:
            result = profile_attention_forward(config, context_length, batch_size)
            if result:
                all_results.append(result)
    
    # Print summary
    if all_results:
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"{'Context Length':<15} {'Batch Size':<12} {'Latency (ms)':<15} {'Throughput':<15} {'TFLOPS':<10}")
        print("-"*80)
        for result in all_results:
            print(f"{result['context_length']:<15} {result['batch_size']:<12} "
                  f"{result['avg_latency_ms']:<15.3f} {result['throughput']:<15.2f} "
                  f"{result['tflops']:<10.2f}")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("No successful profiling results.")
        print("All configurations failed. Check the errors above.")
        print("="*80)


if __name__ == "__main__":
    main()
