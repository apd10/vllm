#!/usr/bin/env python3
"""
Example script demonstrating SKLT sparse attention backend.

This script shows how to use the SKLT backend with streaming indexer
for efficient sparse attention with sink tokens and local window.
"""

from vllm import LLM, SamplingParams
from vllm.config.attention import AttentionConfig, IndexerConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum

# Configure SKLT sparse attention
indexer_config = IndexerConfig(
    indexer_type="streaming",
    num_sink_tokens=4,        # Keep first 4 tokens (attention sink)
    local_window_size=512,    # Local attention window of 512 tokens
    max_sparse_k=1024,        # Maximum sparse keys per head
)

attention_config = AttentionConfig(
    backend=AttentionBackendEnum.SKLT,
    indexer_config=indexer_config,
    use_sklt_sparse_attention=True,
)

print("=" * 70)
print("SKLT Sparse Attention Backend Example")
print("=" * 70)
print(f"\nConfiguration:")
print(f"  Backend: {attention_config.backend.name}")
print(f"  Indexer: {indexer_config.indexer_type}")
print(f"  Sink tokens: {indexer_config.num_sink_tokens}")
print(f"  Local window: {indexer_config.local_window_size}")
print(f"  Max sparse k: {indexer_config.max_sparse_k}")
print()

# Initialize LLM with SKLT backend
print("Initializing LLM with SKLT backend...")
llm = LLM(
    model="facebook/opt-125m",  # Small model for testing
    attention_config=attention_config,
    max_model_len=1024,
    enforce_eager=True,  # Disable CUDA graphs for simplicity
)

print("✓ LLM initialized successfully!")
print()

# Generate text
prompts = [
    "Hello, my name is",
    "The capital of France is",
    "The meaning of life is",
]

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=50,
)

print("Generating responses...")
print("-" * 70)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated_text}")
    print("-" * 70)

print("\n✅ SKLT backend demonstration complete!")
print("\nNote: SKLT uses sparse attention with sink tokens and local window,")
print("      which can be more efficient for long sequences than full attention.")
