# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PyTorch implementation of SKLT sparse attention."""

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.v1.attention.backends.sklt.indexer.base_indexer import SparsityInfo

logger = init_logger(__name__)


def sklt_sparse_attention(
    query: torch.Tensor,           # (num_tokens, num_heads, head_size)
    key_cache: torch.Tensor,       # (num_blocks, block_size, num_kv_heads, head_size)
    value_cache: torch.Tensor,     # (num_blocks, block_size, num_kv_heads, head_size)
    sparsity_info: SparsityInfo,
    block_table: torch.Tensor,     # (batch_size, max_blocks)
    query_start_loc: torch.Tensor, # (batch_size + 1,)
    seq_lens: torch.Tensor,        # (batch_size,)
    block_size: int,
    scale: float,
    num_kv_heads: int,
    output: torch.Tensor,          # Pre-allocated output (num_tokens, num_heads, head_size)
) -> torch.Tensor:
    """Compute sparse attention with per-head sparse patterns.
    
    Pure PyTorch reference implementation for correctness testing.
    This will be slow but correct - optimization comes later.
    
    Args:
        query: Query tensor (num_tokens, num_heads, head_size)
        key_cache: Key cache (num_blocks, block_size, num_kv_heads, head_size)
        value_cache: Value cache (num_blocks, block_size, num_kv_heads, head_size)
        sparsity_info: Sparse pattern information
        block_table: Block table for paged attention
        query_start_loc: Starting position of queries for each sequence
        seq_lens: Sequence lengths
        block_size: Block size for paged KV cache
        scale: Attention scale factor (1/sqrt(head_size))
        num_kv_heads: Number of KV heads (for GQA)
        output: Pre-allocated output tensor
        
    Returns:
        Attention output tensor (same shape as query)
    """
    batch_size = query_start_loc.shape[0] - 1
    num_heads = query.shape[1]
    head_size = query.shape[2]
    num_heads_per_kv = num_heads // num_kv_heads
    
    if query.numel() == 0:
        return output

    query_start_loc = query_start_loc.to(query.device)
    block_table = block_table.to(query.device)
    num_tokens = query.shape[0]

    if torch.cuda.is_current_stream_capturing():
        # CUDA graph capture path: uniform single-token decode.
        # Use direct batch ids to avoid unsupported ops in capture.
        batch_ids = torch.arange(num_tokens, device=query.device)
        q_idx = torch.zeros(num_tokens, device=query.device, dtype=torch.int64)
    else:
        num_q = query_start_loc[1:] - query_start_loc[:-1]  # (B,)
        if num_q.sum().item() != num_tokens:
            raise RuntimeError(
                f"query_start_loc total ({int(num_q.sum().item())}) "
                f"does not match num_tokens ({num_tokens})."
            )
        batch_ids = torch.repeat_interleave(
            torch.arange(batch_size, device=query.device),
            num_q.to(torch.int64),
        )
        token_pos = torch.arange(num_tokens, device=query.device)
        q_idx = token_pos - query_start_loc[batch_ids]

    q_vec = query  # (T, H, D)
    kmax = sparsity_info.sparse_idx.shape[-1]
    pos = torch.arange(kmax, device=query.device).view(1, -1)
    block_table_per = block_table[batch_ids]  # (T, max_blocks)

    for h in range(num_heads):
        sparse_len = sparsity_info.sparse_len[batch_ids, q_idx, h, 0]  # (T,)
        sparse_idx = sparsity_info.sparse_idx[batch_ids, q_idx, h]  # (T, K)
        mask = pos < sparse_len.unsqueeze(1)

        idx = torch.where(mask, sparse_idx, torch.zeros_like(sparse_idx))
        block_idx = idx // block_size
        block_offset = idx % block_size

        physical_block = torch.gather(block_table_per, 1, block_idx)
        kv_head = h // num_heads_per_kv
        k_sparse = key_cache[physical_block, block_offset, kv_head]  # (T, K, D)
        v_sparse = value_cache[physical_block, block_offset, kv_head]  # (T, K, D)

        scores = (k_sparse * q_vec[:, h].unsqueeze(1)).sum(-1) * scale  # (T, K)
        if sparsity_info.sparse_weights is not None:
            sparse_weights = sparsity_info.sparse_weights[batch_ids, q_idx, h]
            scores = scores * sparse_weights.to(scores.dtype)

        scores = scores.masked_fill(~mask, -1e9)
        attn_weights = F.softmax(scores, dim=-1)

        has_keys = sparse_len > 0
        attn_weights = torch.where(
            has_keys.unsqueeze(1),
            attn_weights,
            torch.zeros_like(attn_weights),
        )

        out_vec = torch.einsum("tk,tkd->td", attn_weights, v_sparse)
        output[:num_tokens, h] = out_vec
    
    return output
