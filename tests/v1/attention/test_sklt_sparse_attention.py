# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test for SKLT sparse attention op."""

import pytest
import torch

from vllm.v1.attention.backends.sklt.indexer.base_indexer import SparsityInfo
from vllm.v1.attention.backends.sklt.ops.sparse_attention import (
    sklt_sparse_attention,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_sklt_sparse_attention_matches_manual():
    device = torch.device("cuda")
    batch_size = 1
    num_tokens = 1
    num_heads = 2
    num_kv_heads = 1
    head_size = 2
    block_size = 4
    scale = 1.0

    # Query: (num_tokens, num_heads, head_size)
    query = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]]],
        device=device,
    )

    # KV cache: (num_blocks, block_size, num_kv_heads, head_size) -> 16 keys total.
    key_cache = torch.arange(
        4 * block_size * num_kv_heads * head_size,
        device=device,
        dtype=torch.float32,
    ).reshape(4, block_size, num_kv_heads, head_size)
    value_cache = (
        torch.arange(
            4 * block_size * num_kv_heads * head_size,
            device=device,
            dtype=torch.float32,
        )
        .reshape(4, block_size, num_kv_heads, head_size)
        .add(100.0)
    )

    # Sparse pattern:
    # head 0 -> [0, 2, 4, 8] (4 tokens)
    # head 1 -> [1, 3, 5, 7, 9] (5 tokens)
    max_k = 5
    sparse_len = torch.tensor([[[[4], [5]]]], device=device, dtype=torch.int32)
    sparse_idx = torch.tensor(
        [[[[0, 2, 4, 8, -1], [1, 3, 5, 7, 9]]]],
        device=device,
        dtype=torch.int32,
    )
    sparse_weights = torch.ones((batch_size, 1, num_heads, max_k), device=device)
    sparsity_info = SparsityInfo(
        sparse_len=sparse_len,
        sparse_idx=sparse_idx,
        sparse_weights=sparse_weights,
    )

    block_table = torch.tensor([[0, 1, 2, 3]], device=device, dtype=torch.int32)
    query_start_loc = torch.tensor([0, 1], device=device, dtype=torch.int32)
    seq_lens = torch.tensor([16], device=device, dtype=torch.int32)

    output = torch.empty((num_tokens, num_heads, head_size), device=device)
    sklt_sparse_attention(
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

    expected = torch.empty_like(output)
    for h in range(num_heads):
        k_len = int(sparse_len[0, 0, h, 0].item())
        indices = sparse_idx[0, 0, h, :k_len].tolist()
        k_sparse = torch.stack(
            [
                key_cache[idx // block_size, idx % block_size, 0]
                for idx in indices
            ]
        )
        v_sparse = torch.stack(
            [
                value_cache[idx // block_size, idx % block_size, 0]
                for idx in indices
            ]
        )
        scores = torch.matmul(query[0, h].unsqueeze(0), k_sparse.T).squeeze(0)
        attn = torch.softmax(scores, dim=0)
        expected[0, h] = torch.matmul(attn.unsqueeze(0), v_sparse).squeeze(0)

    torch.testing.assert_close(output, expected)
