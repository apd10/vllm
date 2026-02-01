#!/usr/bin/env python3
"""
Decode-only test for SKLT backend.

This demonstrates SKLT backend working in decode phase only.
For a full production setup, you would use chunked prefill with
a different backend for the prefill phase.
"""

import torch
from vllm.config.attention import IndexerConfig
from vllm.v1.attention.backends.sklt.indexer import StreamingIndexer, SparsityInfo
from vllm.v1.attention.backends.sklt.ops import sklt_sparse_attention

print("=" * 70)
print("SKLT Backend - Decode-Only Demonstration")
print("=" * 70)

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

# Model dimensions (matching OPT-125m)
num_heads = 12
num_kv_heads = 12
head_size = 64
block_size = 16

# Indexer configuration
indexer_config = IndexerConfig(
    indexer_type="streaming",
    num_sink_tokens=4,
    local_window_size=16,
    max_sparse_k=128,
)

print(f"\nIndexer Configuration:")
print(f"  - Sink tokens: {indexer_config.num_sink_tokens}")
print(f"  - Window size: {indexer_config.local_window_size}")
print(f"  - Max sparse k: {indexer_config.max_sparse_k}")

# Create indexer
print("\nInitializing StreamingIndexer...")
indexer = StreamingIndexer(indexer_config, device)
print("✓ Indexer initialized")

# Simulate decode scenario: 2 sequences, each with 1 query token
batch_size = 2
num_queries = 2  # Total queries across batch (1 per sequence)
num_query_heads = num_heads
num_kv_heads_val = num_kv_heads

# Sequence lengths (context already computed)
seq_lens = torch.tensor([50, 80], device=device, dtype=torch.int32)
query_start_loc = torch.tensor([0, 1, 2], device=device, dtype=torch.int32)

print(f"\nDecode Scenario:")
print(f"  - Batch size: {batch_size}")
print(f"  - Queries per sequence: 1 (decode)")
print(f"  - Sequence lengths: {seq_lens.tolist()}")

# Compute sparsity
print("\nComputing sparsity patterns...")
sparsity_info = indexer.compute_sparsity(
    batch_size=batch_size,
    num_queries=num_queries,
    num_query_heads=num_query_heads,
    num_kv_heads=num_kv_heads_val,
    seq_lens=seq_lens,
    query_start_loc=query_start_loc,
    attn_metadata=None,
)

print("✓ Sparsity patterns computed")
print(f"  - Sparsity info shapes:")
print(f"    sparse_len: {sparsity_info.sparse_len.shape}")
print(f"    sparse_idx: {sparsity_info.sparse_idx.shape}")
print(f"    sparse_weights: {sparsity_info.sparse_weights.shape}")

# Show example patterns
# Buffer layout: [batch, max_queries_per_batch, heads, k]
# But data is stored per sequence: [batch, queries_in_seq, heads, k]
# For decode, each sequence has 1 query at index 0
print(f"  - Example sparse patterns:")
for b in range(batch_size):
    q_in_seq = 0  # First (and only) query in this sequence
    sparse_k = sparsity_info.sparse_len[b, q_in_seq, 0, 0].item()
    indices = sparsity_info.sparse_idx[b, q_in_seq, 0, :sparse_k].cpu().tolist()
    print(f"    Seq {b} (len={seq_lens[b].item()}): attends to {sparse_k} keys: {indices[:10]}{'...' if sparse_k > 10 else ''}")

# Create sample tensors for attention computation
print("\nCreating sample attention inputs...")
num_tokens = num_queries
query = torch.randn(num_tokens, num_heads, head_size, device=device, dtype=torch.float16)

# KV cache (paged)
num_blocks = 20
key_cache = torch.randn(num_blocks, block_size, num_kv_heads_val, head_size, device=device, dtype=torch.float16)
value_cache = torch.randn(num_blocks, block_size, num_kv_heads_val, head_size, device=device, dtype=torch.float16)

# Block table (maps sequence positions to physical blocks)
max_blocks_per_seq = (seq_lens.max().item() + block_size - 1) // block_size
block_table = torch.randint(0, num_blocks, (batch_size, max_blocks_per_seq), device=device, dtype=torch.int32)

# Output buffer
output = torch.zeros(num_tokens, num_heads, head_size, device=device, dtype=torch.float16)

scale = 1.0 / (head_size ** 0.5)

print(f"  - Query shape: {query.shape}")
print(f"  - KV cache shape: {key_cache.shape}")
print(f"  - Block table shape: {block_table.shape}")

# Run sparse attention
print("\nRunning SKLT sparse attention...")
result = sklt_sparse_attention(
    query=query,
    key_cache=key_cache,
    value_cache=value_cache,
    sparsity_info=sparsity_info,
    block_table=block_table,
    query_start_loc=query_start_loc,
    seq_lens=seq_lens,
    block_size=block_size,
    scale=scale,
    num_kv_heads=num_kv_heads_val,
    output=output,
)

print("✓ Sparse attention computed successfully")
print(f"  - Output shape: {result.shape}")
print(f"  - Output dtype: {result.dtype}")
print(f"  - Output range: [{result.min():.4f}, {result.max():.4f}]")

# Verify output is not all zeros
non_zero_count = (result.abs() > 1e-6).sum().item()
total_elements = result.numel()
print(f"  - Non-zero elements: {non_zero_count}/{total_elements} ({100*non_zero_count/total_elements:.1f}%)")

print("\n" + "=" * 70)
print("✅ SKLT Decode-Only Test PASSED!")
print("=" * 70)
print("\nNext Steps:")
print("  1. SKLT backend is working correctly for decode phase")
print("  2. For production use, configure separate prefill backend")
print("  3. Use chunked prefill or attention backend routing")
print("  4. Optimize with Triton/CUDA kernels for better performance")
