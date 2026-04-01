# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Abstract base class for SkyLight indexer compute backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sparse_attention_hub.sparse_attention.research_attention.base import (
        ResearchAttentionConfig,
    )

    from vllm.skylight.attention_backend import SkyLightIndexerMetadata
    from vllm.skylight.buffers import SparseIndexBuffers


class SkyLightIndexerBackend(ABC):
    """Abstract compute backend for SkyLight indexer.

    Defines the four operations an indexer must support: prefill/decode
    for both the indexer cache (``kcache_*``) and the index selection
    (``indexer_*``).

    Attributes:
        config: Sparse attention config (contains masker configs).
    """

    def __init__(self, config: ResearchAttentionConfig) -> None:
        self.config: ResearchAttentionConfig = config

    @abstractmethod
    def indexer_prefill(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        metadata: SkyLightIndexerMetadata,
        sparse_index_buffers: SparseIndexBuffers,
        layer_idx: int,
    ) -> None:
        """Compute sparsity pattern for prefill tokens.

        Gathers keys from ``kv_cache`` using ``metadata.block_table`` /
        ``metadata.seq_lens``, scores them against ``q``, and selects top-k.
        Results are written into ``sparse_index_buffers``.

        Args:
            q: Query vectors ``[num_prefill_tokens, n_heads, head_dim]``.
            kv_cache: Paged indexer key cache.
            metadata: Batch info (block_table, seq_lens, etc.).
            sparse_index_buffers: Output buffers — indices, weights, lens.
            layer_idx: Transformer layer index (for per-layer PQ state).
        """
        ...

    @abstractmethod
    def indexer_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        metadata: SkyLightIndexerMetadata,
        sparse_index_buffers: SparseIndexBuffers,
        layer_idx: int,
    ) -> None:
        """Compute sparsity pattern for decode tokens.

        Scores the query against the full key history and selects top-k.
        Results are written into ``sparse_index_buffers``.

        Args:
            q: Query vectors ``[num_decode_tokens, n_heads, head_dim]``.
            kv_cache: Paged indexer key cache.
            metadata: Batch info.
            sparse_index_buffers: Output buffers.
            layer_idx: Transformer layer index.
        """
        ...

    @abstractmethod
    def kcache_prefill(
        self,
        k: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        attn_kv_cache: torch.Tensor,
        attn_slot_mapping: torch.Tensor,
        metadata: SkyLightIndexerMetadata,
        layer_idx: int,
    ) -> None:
        """Build the indexer cache for prefill tokens.

        Reads keys from the **attention** KV cache (``attn_kv_cache``),
        quantises them via PQ (kmeans on first call, incremental assignment
        on subsequent calls), and writes codebook entries into the
        **indexer** cache (``kv_cache``).

        Args:
            k: Key vectors ``[num_prefill_tokens, head_dim]``.
            kv_cache: Paged **indexer** cache tensor (write target).
            slot_mapping: Target slots in the indexer cache.
            attn_kv_cache: Paged **attention** KV cache (read source).
            attn_slot_mapping: Slot mapping for the attention cache.
            metadata: Batch info (block_table, seq_lens for gathering).
            layer_idx: Transformer layer index (for per-layer centroids).
        """
        ...

    @abstractmethod
    def kcache_decode(
        self,
        k: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        attn_kv_cache: torch.Tensor,
        attn_slot_mapping: torch.Tensor,
        metadata: SkyLightIndexerMetadata,
        layer_idx: int,
    ) -> None:
        """Build the indexer cache for decode tokens.

        Reads the newly inserted key from the **attention** KV cache,
        quantises it using centroids built during ``kcache_prefill``, and
        writes the codebook entry into the **indexer** cache.

        Args:
            k: Key vectors ``[num_decode_tokens, head_dim]``.
            kv_cache: Paged **indexer** cache tensor (write target).
            slot_mapping: Target slots in the indexer cache.
            attn_kv_cache: Paged **attention** KV cache (read source).
            attn_slot_mapping: Slot mapping for the attention cache.
            metadata: Batch info.
            layer_idx: Transformer layer index.
        """
        ...
