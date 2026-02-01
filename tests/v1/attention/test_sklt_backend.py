# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for SKLT sparse attention backend."""

import pytest
import torch

from vllm.config.attention import IndexerConfig
from vllm.v1.attention.backends.sklt.indexer import (
    SparsityInfo,
    StreamingIndexer,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum


class TestSKLTBackend:
    """Test SKLT backend registration and basic functionality."""
    
    def test_backend_registration(self):
        """Test that SKLT backend is registered correctly."""
        # Check SKLT is in enum
        assert hasattr(AttentionBackendEnum, "SKLT")
        
        # Get backend class
        backend_cls = AttentionBackendEnum.SKLT.get_class()
        assert backend_cls.get_name() == "SKLT"
        assert backend_cls.is_sparse() is True
    
    def test_sparsity_info_shapes(self):
        """Test SparsityInfo shape validation."""
        device = torch.device("cpu")
        B, Q, H, K = 2, 4, 8, 16
        
        # Valid sparsity info
        sparse_len = torch.zeros((B, Q, H, 1), dtype=torch.int32, device=device)
        sparse_idx = torch.zeros((B, Q, H, K), dtype=torch.int32, device=device)
        sparse_weights = torch.ones((B, Q, H, K), dtype=torch.float32, device=device)
        
        sparsity_info = SparsityInfo(
            sparse_len=sparse_len,
            sparse_idx=sparse_idx,
            sparse_weights=sparse_weights,
        )
        
        # Should not raise
        sparsity_info.validate_shapes()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_streaming_indexer_basic(self):
        """Test StreamingIndexer basic functionality."""
        device = torch.device("cuda")
        
        config = IndexerConfig(
            indexer_type="streaming",
            num_sink_tokens=4,
            local_window_size=8,
            max_sparse_k=128,
        )
        
        indexer = StreamingIndexer(config, device)
        
        # Simple batch with 2 sequences
        batch_size = 2
        num_queries = 4
        num_query_heads = 8
        num_kv_heads = 8
        
        seq_lens = torch.tensor([20, 30], device=device)
        query_start_loc = torch.tensor([0, 2, 4], device=device)
        
        sparsity_info = indexer.compute_sparsity(
            batch_size=batch_size,
            num_queries=num_queries,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            attn_metadata=None,
        )
        
        # Validate shapes
        sparsity_info.validate_shapes()
        assert sparsity_info.sparse_len.shape == (batch_size, num_queries, num_query_heads, 1)
        
        # Check that sparsity lengths are reasonable
        max_expected = config.num_sink_tokens + config.local_window_size
        assert (sparsity_info.sparse_len <= max_expected).all()
        assert (sparsity_info.sparse_len >= 0).all()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_streaming_indexer_pattern(self):
        """Test that streaming indexer produces correct sink + window pattern."""
        device = torch.device("cuda")
        
        config = IndexerConfig(
            indexer_type="streaming",
            num_sink_tokens=2,
            local_window_size=3,
            max_sparse_k=128,
        )
        
        indexer = StreamingIndexer(config, device)
        
        # Single sequence, single query
        batch_size = 1
        num_queries = 1
        num_query_heads = 1
        num_kv_heads = 1
        
        # Sequence length 10, query at position 8 (ctx_len=9, q_idx=0)
        seq_lens = torch.tensor([10], device=device)
        query_start_loc = torch.tensor([0, 1], device=device)
        
        sparsity_info = indexer.compute_sparsity(
            batch_size=batch_size,
            num_queries=num_queries,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            attn_metadata=None,
        )
        
        # Extract pattern for first (only) query
        sparse_k = sparsity_info.sparse_len[0, 0, 0, 0].item()
        indices = sparsity_info.sparse_idx[0, 0, 0, :sparse_k].cpu().tolist()
        weights = sparsity_info.sparse_weights[0, 0, 0, :sparse_k].cpu().tolist()
        
        # Expected pattern for query at position 9:
        # Sink: [0, 1]
        # Window: [6, 7, 8, 9] (last 3 tokens before query + query itself)
        # Union: [0, 1, 6, 7, 8, 9]
        expected_indices = [0, 1, 6, 7, 8, 9]
        
        assert indices == expected_indices, f"Got {indices}, expected {expected_indices}"
        
        # All weights should be 1.0 (uniform)
        assert all(w == 1.0 for w in weights)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
