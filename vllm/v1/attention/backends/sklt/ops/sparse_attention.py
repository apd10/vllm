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
    
    # Move to CPU for indexing operations
    query_start_loc_cpu = query_start_loc.cpu()
    seq_lens_cpu = seq_lens.cpu()
    block_table_cpu = block_table.cpu()
    
    # Process each sequence in batch
    for b in range(batch_size):
        q_start = query_start_loc_cpu[b].item()
        q_end = query_start_loc_cpu[b + 1].item()
        num_q = q_end - q_start
        
        if num_q == 0:
            continue
        
        # Process each query token
        for q_idx in range(num_q):
            q_token_idx = q_start + q_idx
            q_vec = query[q_token_idx]  # (num_heads, head_size)
            
            # Process each query head
            for h in range(num_heads):
                # Get sparse pattern for this query head
                sparse_k = sparsity_info.sparse_len[b, q_idx, h, 0].item()
                
                if sparse_k == 0:
                    # No keys to attend to, output zeros
                    output[q_token_idx, h] = 0.0
                    continue
                
                indices = sparsity_info.sparse_idx[b, q_idx, h, :sparse_k]
                weights = sparsity_info.sparse_weights[b, q_idx, h, :sparse_k]
                
                # Determine KV head for this query head (GQA)
                kv_head = h // num_heads_per_kv
                
                # Gather keys and values from KV cache using indices
                k_sparse_list = []
                v_sparse_list = []
                
                for idx in indices:
                    idx_val = idx.item()
                    
                    # Convert sequence position to block coordinates
                    block_idx = idx_val // block_size
                    block_offset = idx_val % block_size
                    
                    # Lookup physical block in block table
                    physical_block = block_table_cpu[b, block_idx].item()
                    
                    # Extract key and value from cache
                    k_vec = key_cache[physical_block, block_offset, kv_head]  # (head_size,)
                    v_vec = value_cache[physical_block, block_offset, kv_head]  # (head_size,)
                    
                    k_sparse_list.append(k_vec)
                    v_sparse_list.append(v_vec)
                
                # Stack into tensors
                k_sparse = torch.stack(k_sparse_list)  # (sparse_k, head_size)
                v_sparse = torch.stack(v_sparse_list)  # (sparse_k, head_size)
                
                # Compute attention scores: Q @ K^T
                scores = torch.matmul(
                    q_vec[h].unsqueeze(0),  # (1, head_size)
                    k_sparse.transpose(0, 1)  # (head_size, sparse_k)
                ) * scale  # (1, sparse_k)
                
                scores = scores.squeeze(0)  # (sparse_k,)
                
                # Apply sparse weights (element-wise multiply)
                # This allows for weighted attention, not just binary masks
                # Convert weights to match scores dtype
                weights = weights.to(scores.dtype)
                scores = scores * weights
                
                # Softmax over sparse keys
                attn_weights = F.softmax(scores, dim=0)  # (sparse_k,)
                
                # Weighted sum of values
                out_vec = torch.matmul(
                    attn_weights.unsqueeze(0),  # (1, sparse_k)
                    v_sparse  # (sparse_k, head_size)
                )  # (1, head_size)
                
                output[q_token_idx, h] = out_vec.squeeze(0)
    
    return output
