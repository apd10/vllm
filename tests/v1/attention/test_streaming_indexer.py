# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the StreamingIndexer sparsity pattern."""

import pytest
import torch

from vllm.config.attention import IndexerConfig
from vllm.v1.attention.backends.sklt.indexer import StreamingIndexer


def _run_phase1(
    config: IndexerConfig,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_query_heads: int = 2,
    num_kv_heads: int = 1,
):
    indexer = StreamingIndexer(config, seq_lens.device)
    batch_size = seq_lens.numel()
    num_queries = int(query_start_loc[-1].item())
    sparsity_info = indexer.compute_phase1_sparsity(
        batch_size=batch_size,
        num_queries=num_queries,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        attn_metadata=None,
    )
    return sparsity_info


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_streaming_indexer_pattern_cpu():
    device = torch.device("cuda")
    config = IndexerConfig(
        indexer_type="streaming",
        num_sink_tokens=2,
        local_window_size=3,
        max_sparse_k=16,
    )
    seq_lens = torch.tensor([8, 3], device=device)
    query_start_loc = torch.tensor([0, 1, 2], device=device)

    sparsity_info = _run_phase1(
        config,
        seq_lens,
        query_start_loc,
        num_query_heads=2,
        num_kv_heads=1,
    )

    # First sequence: q_pos=7 -> sink [0,1], window [5,6,7]
    for h in range(2):
        sparse_k0 = sparsity_info.sparse_len[0, 0, h, 0].item()
        indices0 = sparsity_info.sparse_idx[0, 0, h, :sparse_k0].tolist()
        assert indices0 == [0, 1, 5, 6, 7]

    # Second sequence: q_pos=2 -> sink [0,1], window [2]
    for h in range(2):
        sparse_k1 = sparsity_info.sparse_len[1, 0, h, 0].item()
        indices1 = sparsity_info.sparse_idx[1, 0, h, :sparse_k1].tolist()
        assert indices1 == [0, 1, 2]

    # Weights should be uniform for all valid indices.
    weights0 = sparsity_info.sparse_weights[0, 0, 0, :sparse_k0].tolist()
    assert all(weight == 1.0 for weight in weights0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_streaming_indexer_truncates_to_max_sparse_k():
    device = torch.device("cuda")
    config = IndexerConfig(
        indexer_type="streaming",
        num_sink_tokens=3,
        local_window_size=4,
        max_sparse_k=7,
    )
    seq_lens = torch.tensor([10], device=device)
    query_start_loc = torch.tensor([0, 1], device=device)

    indexer = StreamingIndexer(config, device)
    # Force truncation by lowering max_sparse_k after init.
    indexer.config.max_sparse_k = 3
    sparsity_info = indexer.compute_phase1_sparsity(
        batch_size=seq_lens.numel(),
        num_queries=int(query_start_loc[-1].item()),
        num_query_heads=2,
        num_kv_heads=1,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        attn_metadata=None,
    )

    for h in range(2):
        sparse_k = sparsity_info.sparse_len[0, 0, h, 0].item()
        indices = sparsity_info.sparse_idx[0, 0, h, :sparse_k].tolist()
        assert sparse_k == indexer.config.max_sparse_k
        assert indices == [0, 1, 2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_streaming_indexer_skips_multi_query():
    device = torch.device("cuda")
    config = IndexerConfig(
        indexer_type="streaming",
        num_sink_tokens=2,
        local_window_size=3,
        max_sparse_k=8,
    )
    # Two queries for a single sequence; should be skipped (buffer only supports 1).
    seq_lens = torch.tensor([6], device=device)
    query_start_loc = torch.tensor([0, 2], device=device)

    with pytest.raises(AssertionError, match="only supports decode"):
        _run_phase1(
            config,
            seq_lens,
            query_start_loc,
            num_query_heads=2,
            num_kv_heads=1,
        )
