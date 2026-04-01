# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Reference PyTorch implementation of the PQ indexer backend.

Uses ``sparse_attention_hub`` PQCache logic as the reference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.skylight.backend import SkyLightIndexerBackend

if TYPE_CHECKING:
    from sparse_attention_hub.sparse_attention.research_attention.base import (
        ResearchAttentionConfig,
    )

    from vllm.skylight.attention_backend import SkyLightIndexerMetadata
    from vllm.skylight.buffers import SparseIndexBuffers


class SkyLightPQBackendRef(SkyLightIndexerBackend):
    """Reference PyTorch PQ backend using ``sparse_attention_hub``.

    Attributes:
        config: Inherited from ``SkyLightIndexerBackend``.
        pq_config: Extracted ``PQCacheConfig`` from ``config.masker_configs``.
        pq_centroids: Per-layer PQ centroids.
            Shape ``[bsz, kv_heads, n_subvec, 2^pq_bits, subvec_d]``.
            ``None`` until first kmeans.
        pq_codebook: Per-layer codebooks.
            Shape ``[bsz, n_quantized_keys, kv_heads, n_subvec]``.
            Grows with sequence.
        pq_ip2l2_phi: Per-layer IP-to-L2 augmentation scalars.
            Only used when ``metric == "ip"``.
    """

    def __init__(self, config: ResearchAttentionConfig) -> None:
        super().__init__(config)
        # TODO: extract PQCacheConfig from config.masker_configs
        self.pq_centroids: dict[int, torch.Tensor | None] = {}
        self.pq_codebook: dict[int, torch.Tensor | None] = {}
        self.pq_ip2l2_phi: dict[int, torch.Tensor | None] = {}

    def indexer_prefill(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        metadata: SkyLightIndexerMetadata,
        sparse_index_buffers: SparseIndexBuffers,
        layer_idx: int,
    ) -> None:
        """Not implemented — prefill runs dense attention."""
        raise NotImplementedError

    def indexer_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        metadata: SkyLightIndexerMetadata,
        sparse_index_buffers: SparseIndexBuffers,
        layer_idx: int,
    ) -> None:
        """Compute sparsity pattern for decode via PQ scoring + top-k."""
        raise NotImplementedError

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
        """Build indexer cache for prefill — kmeans + codebook write."""
        raise NotImplementedError

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
        """Build indexer cache for decode — quantise + codebook write."""
        raise NotImplementedError
