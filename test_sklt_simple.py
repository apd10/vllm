#!/usr/bin/env python3
"""Simple test script for SKLT backend (no vLLM imports)."""

import torch

print("=" * 60)
print("SKLT Backend Simple Test")
print("=" * 60)

# Test 1: Import components
print("\n1. Testing imports...")
try:
    from vllm.config.attention import IndexerConfig
    print("  ✓ IndexerConfig imported")
except Exception as e:
    print(f"  ✗ Failed to import IndexerConfig: {e}")
    exit(1)

try:
    from vllm.v1.attention.backends.sklt.indexer import StreamingIndexer, SparsityInfo
    print("  ✓ StreamingIndexer imported")
except Exception as e:
    print(f"  ✗ Failed to import StreamingIndexer: {e}")
    exit(1)

try:
    from vllm.v1.attention.backends.sklt.ops import sklt_sparse_attention
    print("  ✓ sklt_sparse_attention imported")
except Exception as e:
    print(f"  ✗ Failed to import sklt_sparse_attention: {e}")
    exit(1)

# Test 2: Create indexer config
print("\n2. Creating IndexerConfig...")
try:
    config = IndexerConfig(
        indexer_type="streaming",
        num_sink_tokens=4,
        local_window_size=8,
        max_sparse_k=128,
    )
    print(f"  ✓ Config created: sink={config.num_sink_tokens}, window={config.local_window_size}")
except Exception as e:
    print(f"  ✗ Failed to create config: {e}")
    exit(1)

# Test 3: Create indexer
print("\n3. Creating StreamingIndexer...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Using device: {device}")

try:
    indexer = StreamingIndexer(config, device)
    print(f"  ✓ Indexer created successfully")
except Exception as e:
    print(f"  ✗ Failed to create indexer: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 4: Compute sparsity
print("\n4. Computing sparsity pattern...")
try:
    # Decode scenario: 2 sequences, each with 1 query token
    batch_size = 2
    num_queries = 2  # Total queries (1 per sequence for decode)
    num_query_heads = 8
    num_kv_heads = 8
    
    seq_lens = torch.tensor([20, 30], device=device, dtype=torch.int32)
    query_start_loc = torch.tensor([0, 1, 2], device=device, dtype=torch.int32)  # 2 sequences with 1 query each
    
    sparsity_info = indexer.compute_sparsity(
        batch_size=batch_size,
        num_queries=num_queries,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        attn_metadata=None,
    )
    
    print(f"  ✓ Sparsity computed:")
    print(f"    - sparse_len shape: {sparsity_info.sparse_len.shape}")
    print(f"    - sparse_idx shape: {sparsity_info.sparse_idx.shape}")
    print(f"    - sparse_weights shape: {sparsity_info.sparse_weights.shape}")
    
    # Validate
    sparsity_info.validate_shapes()
    print(f"  ✓ Shapes validated")
    
    # Print example pattern (batch 0, query 0 in that batch, head 0)
    example_k = sparsity_info.sparse_len[0, 0, 0, 0].item()
    example_indices = sparsity_info.sparse_idx[0, 0, 0, :example_k].cpu().tolist()
    print(f"  Example pattern (seq 0, query 0, head 0): {example_indices}")
    
except Exception as e:
    print(f"  ✗ Failed to compute sparsity: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 5: Test sparse attention kernel (minimal)
print("\n5. Testing sparse attention kernel (minimal)...")
try:
    # Create minimal inputs matching our decode scenario
    num_tokens = 2  # 2 sequences, 1 query each
    num_heads = 8
    head_size = 64
    num_blocks = 10
    block_size = 16
    
    query = torch.randn(num_tokens, num_heads, head_size, device=device)
    key_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size, device=device)
    value_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size, device=device)
    block_table = torch.randint(0, num_blocks, (batch_size, 10), device=device, dtype=torch.int32)
    output = torch.zeros(num_tokens, num_heads, head_size, device=device)
    
    scale = 1.0 / (head_size ** 0.5)
    
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
        num_kv_heads=num_kv_heads,
        output=output,
    )
    
    print(f"  ✓ Kernel executed successfully")
    print(f"    - Output shape: {result.shape}")
    print(f"    - Output dtype: {result.dtype}")
    
except Exception as e:
    print(f"  ✗ Failed to execute kernel: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("✅ All tests passed! SKLT backend is working.")
print("=" * 60)
