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
    
    For each query position q at sequence position (ctx_len + q):
    - Include first K sink tokens (positions 0 to K-1)
    - Include last W tokens in local window (positions max(K, pos-W) to pos)
    - Total sparse_k = number of unique positions in [sink ∪ window]
    """
    
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
        max_k = self.config.max_sparse_k
        
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
    
    def compute_sparsity(
        self,
        batch_size: int,
        num_queries: int,
        num_query_heads: int,
        num_kv_heads: int,
        seq_lens: torch.Tensor,
        query_start_loc: torch.Tensor,
        attn_metadata: Any,
    ) -> SparsityInfo:
        """Compute streaming attention pattern.
        
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
        
        # Move seq_lens and query_start_loc to CPU for iteration
        seq_lens_cpu = seq_lens.cpu()
        query_start_loc_cpu = query_start_loc.cpu()
        
        # Compute for each sequence in batch
        for b in range(batch_size):
            seq_len = seq_lens_cpu[b].item()
            q_start = query_start_loc_cpu[b].item()
            q_end = query_start_loc_cpu[b + 1].item()
            num_q = q_end - q_start
            
            if num_q == 0:
                continue
            
            # SKLT only supports single-token decode (num_q must be 1)
            # For chunked prefill or multi-token scenarios, skip this sequence
            # (fallback attention will be used in sklt_impl.py)
            if num_q > max_queries:
                logger.warning_once(
                    f"SKLT: Sequence has {num_q} query tokens but buffer only supports {max_queries}. "
                    f"This should only happen during CUDA graph warmup. Skipping sparsity computation.",
                    scope="sklt_multi_query_skip"
                )
                continue
            
            # Context length (already computed tokens)
            ctx_len = seq_len - num_q
            
            for q_idx in range(num_q):
                # Current query position in sequence
                q_pos = ctx_len + q_idx
                
                # Build sparse pattern for this query
                indices = []
                
                # 1. Add sink tokens (first K tokens, up to current position)
                sink_end = min(num_sink, q_pos + 1)
                indices.extend(range(sink_end))
                
                # 2. Add local window (last W tokens before and including query)
                # Start from max(sink_end, q_pos + 1 - window_size) to avoid duplicates with sink
                window_start = max(num_sink, q_pos + 1 - window_size)
                window_end = q_pos + 1  # Include current position
                if window_start < window_end:
                    indices.extend(range(window_start, window_end))
                
                # Remove duplicates and sort (should already be sorted, but be safe)
                indices = sorted(set(indices))
                sparse_k = len(indices)
                
                # Validate we don't exceed buffer size
                if sparse_k > self.config.max_sparse_k:
                    logger.warning(
                        f"Sparse k ({sparse_k}) exceeds max_sparse_k ({self.config.max_sparse_k}). "
                        f"Truncating to max_sparse_k."
                    )
                    indices = indices[:self.config.max_sparse_k]
                    sparse_k = self.config.max_sparse_k
                
                # Convert to tensor
                indices_tensor = torch.tensor(indices, dtype=torch.int32, device=self.device)
                
                # Populate for all query heads (same pattern across heads for now)
                for h in range(num_query_heads):
                    sparse_len[b, q_idx, h, 0] = sparse_k
                    sparse_idx[b, q_idx, h, :sparse_k] = indices_tensor
                    sparse_weights[b, q_idx, h, :sparse_k] = 1.0  # Uniform weights
        
        return SparsityInfo(
            sparse_len=sparse_len,
            sparse_idx=sparse_idx,
            sparse_weights=sparse_weights,
        )
    
    def get_max_sparse_k(self) -> int:
        """Return maximum sparse keys for streaming pattern."""
        # Maximum is sink tokens + window size (if they don't overlap)
        return self.config.num_sink_tokens + self.config.local_window_size
