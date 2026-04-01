# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fast CUDA implementation of the PQ indexer backend (future work)."""

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


class SkyLightPQBackend(SkyLightIndexerBackend):
    """Fast CUDA PQ backend (placeholder — not yet implemented).

    Same interface as ``SkyLightPQBackendRef`` but with CUDA kernel
    implementations replacing the PyTorch reference logic.
    """

    def __init__(self, config: ResearchAttentionConfig) -> None:
        super().__init__(config)

    def indexer_prefill(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        metadata: SkyLightIndexerMetadata,
        sparse_index_buffers: SparseIndexBuffers,
        layer_idx: int,
    ) -> None:
        raise NotImplementedError

    def indexer_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        metadata: SkyLightIndexerMetadata,
        sparse_index_buffers: SparseIndexBuffers,
        layer_idx: int,
    ) -> None:
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
        raise NotImplementedError
