# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Oracle Top-K indexer: sink + local window + top-k attention."""

from typing import Any

import torch
import torch.nn.functional as F

from vllm.config.attention import IndexerConfig
from vllm.logger import init_logger
from vllm.v1.attention.backends.sklt.indexer.base_indexer import (
    BaseIndexer,
    SparsityInfo,
)

logger = init_logger(__name__)


class OracleTopKIndexer(BaseIndexer):
    """Hybrid indexer combining sink + local window + top-k attention.
    
    Uses both phases:
    - Phase 1: Compute sink + local window pattern (query-independent)
    - Phase 2: Add top-k keys based on Q·K scores (query-dependent)
    
    Configuration (via IndexerConfig):
    - num_sink_tokens: Number of initial tokens to always attend to
    - local_window_size: Size of local attention window
    - max_sparse_k: Total budget for sparse keys (sink + window + top-k)
    - top_k: Number of additional top-k keys to select (optional, computed from max_sparse_k)
    
    Example:
        num_sink_tokens=4, local_window_size=32, max_sparse_k=128
        Phase 1: Allocate ~36 keys for sink+window
        Phase 2: Add up to 92 more keys via top-k selection
    """
    
    def needs_phase2_refinement(self) -> bool:
        """Oracle Top-K needs Phase 2 to add top-k keys."""
        return True
    
    def _init_buffers(self):
        """Pre-allocate buffers for both phases."""
        # Same buffer sizing as streaming indexer
        max_batch = 512     # Maximum concurrent sequences
        max_queries = 1     # Decode only: single query token per sequence
        max_heads = 128     # Maximum attention heads
        max_k = self.config.max_sparse_k
        
        logger.info(
            f"OracleTopKIndexer: Allocating buffers for "
            f"batch={max_batch}, queries={max_queries}, heads={max_heads}, "
            f"max_sparse_k={max_k}"
        )
        
        # Phase 1 buffers: sink + window pattern
        self.phase1_len_buffer = torch.zeros(
            (max_batch, max_queries, max_heads, 1),
            dtype=torch.int32,
            device=self.device
        )
        self.phase1_idx_buffer = torch.zeros(
            (max_batch, max_queries, max_heads, max_k),
            dtype=torch.int32,
            device=self.device
        )
        
        # Phase 2 buffers: final pattern (sink + window + top-k)
        self.phase2_len_buffer = torch.zeros(
            (max_batch, max_queries, max_heads, 1),
            dtype=torch.int32,
            device=self.device
        )
        self.phase2_idx_buffer = torch.zeros(
            (max_batch, max_queries, max_heads, max_k),
            dtype=torch.int32,
            device=self.device
        )
        
        # Temporary buffers for top-k computation
        self.scores_buffer = torch.zeros(
            (max_batch, max_queries, max_heads, 8192),  # Max context length for scoring
            dtype=torch.float32,
            device=self.device
        )
        
        logger.info(
            f"OracleTopKIndexer: Buffers allocated successfully. "
            f"Memory: ~{self._estimate_buffer_memory_mb():.2f} MB"
        )
    
    def _estimate_buffer_memory_mb(self) -> float:
        """Estimate buffer memory usage in MB."""
        total_bytes = 0
        for buf in [self.phase1_len_buffer, self.phase1_idx_buffer,
                    self.phase2_len_buffer, self.phase2_idx_buffer,
                    self.scores_buffer]:
            total_bytes += buf.numel() * buf.element_size()
        return total_bytes / (1024 * 1024)
    
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
        """Phase 1: Compute sink + local window pattern.
        
        This creates the base pattern that will be extended with top-k
        in Phase 2.
        """
        # Validate buffer capacity
        max_batch = self.phase1_len_buffer.shape[0]
        max_queries = self.phase1_len_buffer.shape[1]
        
        if batch_size > max_batch:
            raise RuntimeError(
                f"Batch size ({batch_size}) exceeds buffer capacity ({max_batch})"
            )
        
        num_sink = self.config.num_sink_tokens
        window_size = self.config.local_window_size
        
        # Get view of Phase 1 buffers
        sparse_len = self.phase1_len_buffer[:batch_size, :max_queries, :num_query_heads]
        sparse_idx = self.phase1_idx_buffer[:batch_size, :max_queries, :num_query_heads]
        
        # Reset buffers
        sparse_len.fill_(0)
        sparse_idx.fill_(-1)
        
        # Move to CPU for iteration
        seq_lens_cpu = seq_lens.cpu()
        query_start_loc_cpu = query_start_loc.cpu()
        
        # Compute sink + window pattern for each sequence
        for b in range(batch_size):
            seq_len = seq_lens_cpu[b].item()
            q_start = query_start_loc_cpu[b].item()
            q_end = query_start_loc_cpu[b + 1].item()
            num_q = q_end - q_start
            
            if num_q == 0 or num_q > max_queries:
                continue
            
            ctx_len = seq_len - num_q
            
            for q_idx in range(num_q):
                q_pos = ctx_len + q_idx
                
                # Build sink + window pattern
                indices = []
                
                # 1. Sink tokens
                sink_end = min(num_sink, q_pos + 1)
                indices.extend(range(sink_end))
                
                # 2. Local window
                window_start = max(num_sink, q_pos + 1 - window_size)
                window_end = q_pos + 1
                if window_start < window_end:
                    indices.extend(range(window_start, window_end))
                
                # Remove duplicates and sort
                indices = sorted(set(indices))
                phase1_k = len(indices)
                
                # Convert to tensor
                indices_tensor = torch.tensor(indices, dtype=torch.int32, device=self.device)
                
                # Populate for all query heads
                for h in range(num_query_heads):
                    sparse_len[b, q_idx, h, 0] = phase1_k
                    sparse_idx[b, q_idx, h, :phase1_k] = indices_tensor
        
        return SparsityInfo(
            sparse_len=sparse_len,
            sparse_idx=sparse_idx,
            sparse_weights=None,  # No weights in Phase 1
        )
    
    def compute_phase2_sparsity(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        phase1_sparsity: SparsityInfo,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        query_start_loc: torch.Tensor,
        scale: float,
        block_size: int,
    ) -> SparsityInfo:
        """Phase 2: Extend pattern with top-k keys based on Q·K scores.
        
        Takes the sink + window pattern from Phase 1 and adds top-k keys
        from the remaining context based on attention scores.
        """
        # Extract dimensions
        num_tokens, num_heads, head_size = query.shape
        batch_size = seq_lens.shape[0]
        
        # Get Phase 2 buffers
        phase2_len = self.phase2_len_buffer[:batch_size, :1, :num_heads]
        phase2_idx = self.phase2_idx_buffer[:batch_size, :1, :num_heads]
        
        # Reset Phase 2 buffers
        phase2_len.fill_(0)
        phase2_idx.fill_(-1)
        
        # Move to CPU for iteration
        seq_lens_cpu = seq_lens.cpu()
        query_start_loc_cpu = query_start_loc.cpu()
        
        # Compute top-k budget (total budget - phase1 keys)
        max_total_k = self.config.max_sparse_k
        
        for b in range(batch_size):
            seq_len = seq_lens_cpu[b].item()
            q_start = query_start_loc_cpu[b].item()
            q_end = query_start_loc_cpu[b + 1].item()
            num_q = q_end - q_start
            
            if num_q != 1:  # Only support single-query decode
                continue
            
            # Get Phase 1 indices for this sequence
            phase1_k = phase1_sparsity.sparse_len[b, 0, 0, 0].item()
            phase1_indices = phase1_sparsity.sparse_idx[b, 0, 0, :phase1_k].cpu().tolist()
            phase1_set = set(phase1_indices)
            
            # Compute top-k budget
            topk_budget = max_total_k - phase1_k
            if topk_budget <= 0:
                # No room for top-k, just use Phase 1 pattern
                for h in range(num_heads):
                    phase2_len[b, 0, h, 0] = phase1_k
                    phase2_idx[b, 0, h, :phase1_k] = phase1_sparsity.sparse_idx[b, 0, h, :phase1_k]
                continue
            
            # Get query for this sequence
            q_vec = query[q_start:q_end]  # (1, num_heads, head_size)
            
            # Compute Q·K scores for all positions
            # This is a simplified version - in practice, you'd want to:
            # 1. Gather keys from KV cache using block_table
            # 2. Compute scores per head
            # 3. Select top-k per head
            
            # For now, we'll use a placeholder that selects uniformly spaced keys
            # TODO: Implement actual Q·K score computation
            candidate_positions = [i for i in range(seq_len) if i not in phase1_set]
            
            if len(candidate_positions) > 0:
                # Select top-k from candidates (placeholder: uniform sampling)
                num_topk = min(topk_budget, len(candidate_positions))
                step = max(1, len(candidate_positions) // num_topk)
                topk_indices = candidate_positions[::step][:num_topk]
            else:
                topk_indices = []
            
            # Combine Phase 1 and top-k indices
            final_indices = sorted(phase1_indices + topk_indices)
            final_k = len(final_indices)
            
            # Populate Phase 2 buffers
            final_indices_tensor = torch.tensor(final_indices, dtype=torch.int32, device=self.device)
            for h in range(num_heads):
                phase2_len[b, 0, h, 0] = final_k
                phase2_idx[b, 0, h, :final_k] = final_indices_tensor
        
        return SparsityInfo(
            sparse_len=phase2_len,
            sparse_idx=phase2_idx,
            sparse_weights=None,
        )
    
    def get_max_sparse_k(self) -> int:
        """Return maximum sparse keys (total budget)."""
        return self.config.max_sparse_k
