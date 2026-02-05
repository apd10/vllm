# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming indexer for sink tokens + local window pattern."""

from typing import Any

import torch

from vllm.config.attention import IndexerConfig
from vllm.logger import init_logger
from vllm.v1.attention.backends.sklt.indexer.base_indexer import (
    BaseIndexer,
    SparsityInfo,
)

logger = init_logger(__name__)


class StreamingIndexer(BaseIndexer):
    """Indexer for streaming attention with sink tokens and local window.
    
    This is a query-independent indexer - the pattern is fully determined
    in Phase 1 without needing query information. Phase 2 is not needed.
    
    For each query position q at sequence position (ctx_len + q):
    - Include first K sink tokens (positions 0 to K-1)
    - Include last W tokens in local window (positions max(K, pos-W) to pos)
    - Total sparse_k = number of unique positions in [sink ∪ window]
    """
    
    def needs_phase2_refinement(self) -> bool:
        """Streaming indexer doesn't need Phase 2 refinement."""
        return False
    
    def _init_buffers(self):
        """Pre-allocate buffers based on config and scheduler limits."""
        # SKLT backend is decode-only (single query token per sequence)
        # Buffer sizing:
        # - max_queries = 1 because we only support decode (not prefill)
        # - max_batch = maximum number of concurrent sequences (must match CUDA graph sizes)
        # - max_heads = maximum attention heads in the model
        max_batch = 512     # Maximum concurrent sequences (matches max CUDA graph capture size)
        max_queries = 1     # Decode only: single query token per sequence
        max_heads = 128     # Maximum attention heads (to support most models)
        max_k = min(self.config.max_sparse_k, self.get_max_sparse_k())
        
        logger.info(
            f"StreamingIndexer: Allocating buffers for "
            f"batch={max_batch}, queries={max_queries}, heads={max_heads}, "
            f"max_sparse_k={max_k}"
        )
        
        # Allocate on device for CUDA graph compatibility
        self.sparse_len_buffer = torch.zeros(
            (max_batch, max_queries, max_heads, 1),
            dtype=torch.int32,
            device=self.device
        )
        self.sparse_idx_buffer = torch.zeros(
            (max_batch, max_queries, max_heads, max_k),
            dtype=torch.int32,
            device=self.device
        )
        self.sparse_weights_buffer = torch.ones(
            (max_batch, max_queries, max_heads, max_k),
            dtype=torch.float32,
            device=self.device
        )
        
        logger.info(
            f"StreamingIndexer: Buffers allocated successfully. "
            f"Memory: ~{self._estimate_buffer_memory_mb():.2f} MB"
        )
    
    def _estimate_buffer_memory_mb(self) -> float:
        """Estimate buffer memory usage in MB."""
        len_bytes = self.sparse_len_buffer.numel() * self.sparse_len_buffer.element_size()
        idx_bytes = self.sparse_idx_buffer.numel() * self.sparse_idx_buffer.element_size()
        weights_bytes = self.sparse_weights_buffer.numel() * self.sparse_weights_buffer.element_size()
        total_mb = (len_bytes + idx_bytes + weights_bytes) / (1024 * 1024)
        return total_mb
    
    def compute_phase1_sparsity(
        self,
        batch_size: int,
        num_queries: int,
        num_query_heads: int,
        num_kv_heads: int,
        seq_lens: torch.Tensor,
        query_start_loc: torch.Tensor,
        attn_metadata: Any,
    ) -> SparsityInfo:
        """Phase 1: Compute complete streaming attention pattern.
        
        For streaming indexer, the entire pattern is determined here
        (sink + window). No Phase 2 refinement needed.
        
        PyTorch implementation for correctness (to be optimized later).
        """
        # Validate buffer capacity
        max_batch = self.sparse_len_buffer.shape[0]
        max_queries = self.sparse_len_buffer.shape[1]
        
        if batch_size > max_batch:
            raise RuntimeError(
                f"Batch size ({batch_size}) exceeds buffer capacity ({max_batch}). "
                f"This should not happen - please file a bug report."
            )
        
        num_sink = self.config.num_sink_tokens
        window_size = self.config.local_window_size
        
        # Get view of buffers for current batch size
        sparse_len = self.sparse_len_buffer[:batch_size, :max_queries, :num_query_heads]
        sparse_idx = self.sparse_idx_buffer[:batch_size, :max_queries, :num_query_heads]
        sparse_weights = self.sparse_weights_buffer[:batch_size, :max_queries, :num_query_heads]
        
        # Reset buffers
        sparse_len.fill_(0)
        sparse_idx.fill_(-1)  # Invalid index marker
        sparse_weights.fill_(0.0)
        
        # Compute per-sequence query counts on device.
        q_start = query_start_loc[:batch_size]
        q_end = query_start_loc[1:batch_size + 1]
        num_q = q_end - q_start

        # SKLT only supports single-token decode (num_q must be 1).
        # For chunked prefill or multi-token scenarios, skip these sequences.
        if (num_q > max_queries).any():
            logger.warning_once(
                f"SKLT: Sequence has >{max_queries} query tokens but buffer only supports {max_queries}. "
                f"This should only happen during CUDA graph warmup. Skipping sparsity computation.",
                scope="sklt_multi_query_skip",
            )

        # Only decode (0 or 1 query token per sequence) is supported here.
        assert (num_q <= 1).all(), (
            "StreamingIndexer only supports decode; "
            "multi-token queries must use fallback attention."
        )
        valid_mask = num_q == 1
        seq_lens_i32 = seq_lens.to(torch.int32)
        num_q_i32 = num_q.to(torch.int32)
        q_pos = seq_lens_i32 - num_q_i32  # q_idx is always 0 for decode

        sink_end = torch.minimum(
            torch.tensor(num_sink, device=self.device, dtype=torch.int32),
            q_pos + 1,
        )
        window_start = torch.maximum(
            torch.tensor(num_sink, device=self.device, dtype=torch.int32),
            q_pos + 1 - window_size,
        )
        window_end = q_pos + 1

        sink_len = sink_end
        window_len = torch.clamp(window_end - window_start, min=0)
        sparse_k = sink_len + window_len

        buffer_k = sparse_idx.shape[-1]
        max_k = min(self.config.max_sparse_k, buffer_k)
        if (sparse_k > max_k).any():
            logger.warning(
                f"Sparse k ({int(sparse_k.max().item())}) exceeds max_sparse_k ({max_k}). "
                f"Truncating to max_sparse_k."
            )
        sparse_k = torch.minimum(
            sparse_k, torch.tensor(max_k, device=self.device, dtype=torch.int32)
        )
        sparse_k = torch.where(valid_mask, sparse_k, torch.zeros_like(sparse_k))

        # Build indices for each batch in a vectorized manner.
        pos = torch.arange(buffer_k, device=self.device, dtype=torch.int32).unsqueeze(0)
        sink_len_exp = sink_len.unsqueeze(1)
        window_start_exp = window_start.unsqueeze(1)
        sparse_k_exp = sparse_k.unsqueeze(1)
        valid_pos = pos < max_k

        indices = torch.where(
            pos < sink_len_exp,
            pos,
            window_start_exp + (pos - sink_len_exp),
        )
        indices = torch.where(
            valid_pos & (pos < sparse_k_exp),
            indices,
            torch.full_like(indices, -1),
        )

        weights = (valid_pos & (pos < sparse_k_exp)).to(torch.float32)

        # Populate for all query heads (same pattern across heads).
        sparse_len[:, 0, :num_query_heads, 0] = sparse_k.unsqueeze(1).expand(
            -1, num_query_heads
        )
        sparse_idx[:, 0, :num_query_heads, :] = indices.unsqueeze(1).expand(
            -1, num_query_heads, -1
        )
        sparse_weights[:, 0, :num_query_heads, :] = weights.unsqueeze(1).expand(
            -1, num_query_heads, -1
        )
        
        return SparsityInfo(
            sparse_len=sparse_len,
            sparse_idx=sparse_idx,
            sparse_weights=sparse_weights,
        )
    
    def get_max_sparse_k(self) -> int:
        """Return maximum sparse keys for streaming pattern."""
        # Maximum is sink tokens + window size (if they don't overlap)
        return self.config.num_sink_tokens + self.config.local_window_size
