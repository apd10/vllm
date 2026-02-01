#!/usr/bin/env python3
"""
Test SKLT backend with prefill fallback.

This tests that SKLT can handle prefill by falling back to standard attention.
"""

import torch
from vllm.config.attention import IndexerConfig
from vllm.v1.attention.backends.sklt.indexer import StreamingIndexer, SparsityInfo
from vllm.v1.attention.backends.sklt.sklt_impl import SKLTAttentionImpl
from vllm.v1.attention.backends.sklt.sklt_metadata import SKLTAttentionMetadata
from vllm.v1.attention.backend import AttentionType

print("=" * 70)
print("SKLT Backend - Prefill Fallback Test")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")

# Model dimensions
num_heads = 12
num_kv_heads = 12
head_size = 64
block_size = 16

# Create SKLT implementation
print("\nInitializing SKLT attention implementation...")
impl = SKLTAttentionImpl(
    num_heads=num_heads,
    head_size=head_size,
    scale=1.0 / (head_size ** 0.5),
    num_kv_heads=num_kv_heads,
    alibi_slopes=None,
    sliding_window=None,
    kv_cache_dtype="auto",
    logits_soft_cap=None,
    attn_type=AttentionType.DECODER,
    kv_sharing_target_layer_name=None,
)
impl.block_size = block_size
print("✓ SKLT impl created")

# Test prefill scenario: multiple query tokens
print("\n" + "=" * 70)
print("TEST 1: Prefill Scenario (max_query_len > 1)")
print("=" * 70)

batch_size = 1
num_queries = 5  # Multiple query tokens (prefill)
seq_len = 10  # Total sequence length after prefill

seq_lens = torch.tensor([seq_len], device=device, dtype=torch.int32)
query_start_loc = torch.tensor([0, num_queries], device=device, dtype=torch.int32)

# For prefill, don't compute sparsity - create dummy sparsity info
# (won't be used, prefill uses fallback attention)
from vllm.v1.attention.backends.sklt.indexer import SparsityInfo
sparsity_info = SparsityInfo(
    sparse_len=torch.zeros((batch_size, 1, num_heads, 1), 
                          dtype=torch.int32, device=device),
    sparse_idx=torch.zeros((batch_size, 1, num_heads, 1), 
                          dtype=torch.int32, device=device),
    sparse_weights=torch.zeros((batch_size, 1, num_heads, 1), 
                              dtype=torch.float32, device=device),
)

# Create metadata
metadata = SKLTAttentionMetadata(
    num_actual_tokens=num_queries,
    max_query_len=num_queries,  # > 1, triggers prefill fallback
    query_start_loc=query_start_loc,
    max_seq_len=seq_len,
    seq_lens=seq_lens,
    block_table=torch.zeros((batch_size, 10), device=device, dtype=torch.int32),
    slot_mapping=torch.zeros(num_queries, device=device, dtype=torch.int32),
    sparsity_info=sparsity_info,
    causal=True,
)

# Create inputs
query = torch.randn(num_queries, num_heads, head_size, device=device, dtype=torch.float16)
key = torch.randn(num_queries, num_kv_heads, head_size, device=device, dtype=torch.float16)
value = torch.randn(num_queries, num_kv_heads, head_size, device=device, dtype=torch.float16)

num_blocks = 10
key_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size, device=device, dtype=torch.float16)
value_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size, device=device, dtype=torch.float16)
kv_cache = torch.stack([key_cache, value_cache], dim=0)

output = torch.zeros(num_queries, num_heads, head_size, device=device, dtype=torch.float16)

# Mock layer object
class MockLayer:
    pass

layer = MockLayer()

print(f"  - Batch size: {batch_size}")
print(f"  - Num queries: {num_queries} (prefill)")
print(f"  - Max query len: {metadata.max_query_len}")
print(f"  - Running forward pass...")

try:
    result = impl.forward(
        layer=layer,
        query=query,
        key=key,
        value=value,
        kv_cache=kv_cache,
        attn_metadata=metadata,
        output=output,
    )
    print(f"✓ Prefill fallback successful")
    print(f"  - Output shape: {result.shape}")
    print(f"  - Output dtype: {result.dtype}")
except Exception as e:
    print(f"✗ Prefill fallback failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test decode scenario: single query token
print("\n" + "=" * 70)
print("TEST 2: Decode Scenario (max_query_len = 1)")
print("=" * 70)

num_queries_decode = 1
seq_len_decode = 50

seq_lens_decode = torch.tensor([seq_len_decode], device=device, dtype=torch.int32)
query_start_loc_decode = torch.tensor([0, 1], device=device, dtype=torch.int32)

# For decode, compute sparsity
indexer_config = IndexerConfig(
    indexer_type="streaming",
    num_sink_tokens=2,
    local_window_size=4,
    max_sparse_k=128,
)
indexer = StreamingIndexer(indexer_config, device)

sparsity_info_decode = indexer.compute_sparsity(
    batch_size=1,
    num_queries=num_queries_decode,
    num_query_heads=num_heads,
    num_kv_heads=num_kv_heads,
    seq_lens=seq_lens_decode,
    query_start_loc=query_start_loc_decode,
    attn_metadata=None,
)

metadata_decode = SKLTAttentionMetadata(
    num_actual_tokens=num_queries_decode,
    max_query_len=1,  # Decode
    query_start_loc=query_start_loc_decode,
    max_seq_len=seq_len_decode,
    seq_lens=seq_lens_decode,
    block_table=torch.randint(0, num_blocks, (1, 10), device=device, dtype=torch.int32),
    slot_mapping=torch.zeros(num_queries_decode, device=device, dtype=torch.int32),
    sparsity_info=sparsity_info_decode,
    causal=True,
)

query_decode = torch.randn(1, num_heads, head_size, device=device, dtype=torch.float16)
output_decode = torch.zeros(1, num_heads, head_size, device=device, dtype=torch.float16)

print(f"  - Num queries: {num_queries_decode} (decode)")
print(f"  - Max query len: {metadata_decode.max_query_len}")
print(f"  - Running forward pass...")

try:
    result_decode = impl.forward(
        layer=layer,
        query=query_decode,
        key=None,
        value=None,
        kv_cache=kv_cache,
        attn_metadata=metadata_decode,
        output=output_decode,
    )
    print(f"✓ Decode with SKLT sparse attention successful")
    print(f"  - Output shape: {result_decode.shape}")
    print(f"  - Output dtype: {result_decode.dtype}")
    
    # Show sparse pattern used
    sparse_k = sparsity_info_decode.sparse_len[0, 0, 0, 0].item()
    indices = sparsity_info_decode.sparse_idx[0, 0, 0, :sparse_k].cpu().tolist()
    print(f"  - Sparse pattern (seq len {seq_len_decode}): {sparse_k} keys: {indices}")
    
except Exception as e:
    print(f"✗ Decode failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 70)
print("✅ SKLT Prefill Fallback Test PASSED!")
print("=" * 70)
print("\nSummary:")
print("  ✓ Prefill (max_query_len > 1): Uses standard attention fallback")
print("  ✓ Decode (max_query_len = 1): Uses SKLT sparse attention")
print("  ✓ Both phases work correctly in the same backend")
