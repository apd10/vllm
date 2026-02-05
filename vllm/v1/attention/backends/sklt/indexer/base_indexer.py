# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base indexer interface for SKLT sparse attention."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

from vllm.config.attention import IndexerConfig


@dataclass
class SparsityInfo:
    """Container for sparsity information.
    
    Attributes:
        sparse_len: (B, queries, num_query_heads, 1) - number of valid keys per query head
        sparse_idx: (B, queries, num_query_heads, max_sparse_k) - indices of sparse keys
        sparse_weights: (B, queries, num_query_heads, max_sparse_k) - optional attention weights
    """
    sparse_len: torch.Tensor
    sparse_idx: torch.Tensor
    sparse_weights: torch.Tensor | None = None
    
    def validate_shapes(self):
        """Ensure all tensors have compatible shapes."""
        B, Q, H, one = self.sparse_len.shape
        assert one == 1, f"sparse_len last dim should be 1, got {one}"
        
        max_k = self.sparse_idx.shape[-1]
        assert self.sparse_idx.shape == (B, Q, H, max_k), (
            f"sparse_idx shape mismatch: expected {(B, Q, H, max_k)}, "
            f"got {self.sparse_idx.shape}"
        )
        if self.sparse_weights is not None:
            assert self.sparse_weights.shape == (B, Q, H, max_k), (
                f"sparse_weights shape mismatch: expected {(B, Q, H, max_k)}, "
                f"got {self.sparse_weights.shape}"
            )


class BaseIndexer(ABC):
    """Base class for sparse attention indexers.
    
    Two-phase architecture:
    - Phase 1 (compute_phase1_sparsity): Query-independent pattern + buffer allocation
    - Phase 2 (compute_phase2_sparsity): Query-dependent refinement (optional)
    
    Examples:
    - Query-independent (e.g., streaming): Complete pattern in Phase 1, Phase 2 returns unchanged
    - Query-dependent (e.g., pure top-k): Empty pattern in Phase 1, complete pattern in Phase 2
    - Hybrid (e.g., sink+window+top-k): Base pattern in Phase 1, extended pattern in Phase 2
    """
    
    def __init__(self, indexer_config: IndexerConfig, device: torch.device):
        """Initialize indexer.
        
        Args:
            indexer_config: Configuration for the indexer
            device: Device to allocate buffers on
        """
        self.config = indexer_config
        self.device = device
        # Pre-allocate buffers for CUDA graph compatibility
        self._init_buffers()
    
    @abstractmethod
    def _init_buffers(self):
        """Initialize reusable buffers for CUDA graph compatibility.
        
        Buffers should be allocated at maximum size based on config
        and scheduler limits. Actual usage will slice these buffers.
        """
        pass
    
    def needs_phase2_refinement(self) -> bool:
        """Check if this indexer needs Phase 2 refinement.
        
        Returns:
            True if Phase 2 will modify the sparsity pattern, False otherwise.
            
        Default: False (query-independent, Phase 2 not needed)
        Override to return True for query-dependent or hybrid indexers.
        """
        return False
    
    @abstractmethod
    def compute_phase1_sparsity(
        self,
        batch_size: int,
        num_queries: int,
        num_query_heads: int,
        num_kv_heads: int,
        seq_lens: torch.Tensor,  # (B,) current sequence lengths
        query_start_loc: torch.Tensor,  # (B+1,) query positions
        attn_metadata: Any,  # Backend-specific metadata
    ) -> SparsityInfo:
        """Phase 1: Compute query-independent sparsity pattern.
        
        Called in metadata builder (before model forward).
        
        For query-independent indexers: Returns complete sparsity pattern
        For query-dependent indexers: Returns empty/placeholder pattern with allocated buffers
        For hybrid indexers: Returns base pattern (e.g., sink+window) to be extended in Phase 2
        
        Args:
            batch_size: Number of sequences in batch
            num_queries: Total number of query tokens across batch
            num_query_heads: Number of query attention heads
            num_kv_heads: Number of key/value heads (for GQA)
            seq_lens: Sequence lengths for each request in batch
            query_start_loc: Starting position of queries for each request
            attn_metadata: Additional metadata (backend-specific)
            
        Returns:
            SparsityInfo with base pattern (may be extended in Phase 2)
        """
        pass
    
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
        """Phase 2: Refine/extend sparsity pattern using query information.
        
        Called in attention forward (after Q projection is computed).
        Only called if needs_phase2_refinement() returns True.
        
        For query-independent indexers: Not called (needs_phase2_refinement() = False)
        For query-dependent indexers: Computes complete pattern using query
        For hybrid indexers: Extends phase1_sparsity with query-dependent indices
        
        Args:
            query: Query tensor (num_tokens, num_heads, head_size)
            key_cache: Key cache (num_blocks, block_size, num_kv_heads, head_size)
            phase1_sparsity: Sparsity info from Phase 1
            block_table: Block table mapping (batch_size, max_blocks)
            seq_lens: Sequence lengths (batch_size,)
            query_start_loc: Query start locations (batch_size + 1,)
            scale: Attention scale factor
            block_size: KV cache block size
            
        Returns:
            Final sparsity info (may be same as phase1_sparsity or extended)
        """
        # Default: return Phase 1 sparsity unchanged (for QUERY_INDEPENDENT)
        return phase1_sparsity
    
    @abstractmethod
    def get_max_sparse_k(self) -> int:
        """Return maximum number of sparse keys per head.
        
        Returns:
            Maximum sparse keys that could be generated by this indexer
        """
        pass
