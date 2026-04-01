# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Top-level SkyLightIndexer module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from vllm.forward_context import get_forward_context
from vllm.skylight.backend import SkyLightIndexerBackend
from vllm.skylight.buffers import SparseIndexBuffers
from vllm.skylight.cache import SkyLightIndexerCache

if TYPE_CHECKING:
    from sparse_attention_hub.sparse_attention.research_attention.base import (
        ResearchAttentionConfig,
    )

    from vllm.config import VllmConfig
    from vllm.skylight.attention_backend import SkyLightIndexerMetadata


class SkyLightIndexer(nn.Module):
    """Top-level indexer module instantiated by model code.

    Orchestrates the indexer cache and pluggable compute backend.  Called
    once per transformer layer during forward.

    Attributes:
        vllm_config: Global vLLM config.
        sparse_attention_config: Sparse attention config from
            ``sparse_attention_hub``.
        backend: Compute backend instance (created from ``backend_cls``).
        cache: Indexer cache instance (manages paged KV cache).
        sparse_index_buffers: Shared output buffers.
        prefix: Layer name prefix.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        sparse_attention_config: ResearchAttentionConfig,
        backend_cls: type[SkyLightIndexerBackend],
        cache: SkyLightIndexerCache,
        sparse_index_buffers: SparseIndexBuffers,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.vllm_config: VllmConfig = vllm_config
        self.sparse_attention_config: ResearchAttentionConfig = (
            sparse_attention_config
        )
        self.backend: SkyLightIndexerBackend = backend_cls(
            sparse_attention_config
        )
        self.cache: SkyLightIndexerCache = cache
        self.sparse_index_buffers: SparseIndexBuffers = sparse_index_buffers
        self.prefix: str = prefix

    def forward(
        self,
        hidden_states: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        layer_idx: int,
    ) -> SparseIndexBuffers:
        """Run indexer for a single transformer layer.

        Args:
            hidden_states: Layer input ``[num_tokens, hidden_size]``
                (unused by indexer, passed for interface compatibility).
            q: Query vectors ``[num_tokens, n_heads, head_dim]``
                (after projection + RoPE).
            k: Key vectors ``[num_tokens, head_dim]``
                (after projection + RoPE).
            layer_idx: Which transformer layer.

        Returns:
            The ``SparseIndexBuffers`` populated with the sparsity pattern
            for this layer.
        """
        raise NotImplementedError
