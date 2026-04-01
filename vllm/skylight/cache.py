# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""SkyLight indexer cache classes.

Contains:
- ``SkyLightIndexerCache`` — base cache (``AttentionLayerBase`` integration).
- ``SkyLightIndexerPQCache`` — PQ-specific subclass returning ``PQCacheSpec``.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import torch
from torch import nn

from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.skylight.attention_backend import SkyLightAttentionBackend
from vllm.skylight.kv_cache_spec import PQCacheSpec
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import KVCacheSpec

if TYPE_CHECKING:
    pass


class SkyLightIndexerCache(nn.Module, AttentionLayerBase):
    """Base class for SkyLight indexer caches.

    Implements ``AttentionLayerBase`` so that the vLLM framework discovers
    the layer, groups it, and allocates paged cache blocks for it.

    Attributes:
        prefix: Unique layer name
            (e.g. ``"model.layers.0.self_attn.indexer.k_cache"``).
        cache_config: Block size and cache config from vLLM.
        kv_cache: Paged cache tensor — initialised empty, populated by
            the worker via ``bind_kv_cache()``.
    """

    def __init__(self, prefix: str, cache_config: CacheConfig) -> None:
        super().__init__()
        self.prefix: str = prefix
        self.cache_config: CacheConfig = cache_config
        self.kv_cache: torch.Tensor = torch.tensor([])

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    @abstractmethod
    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        """Returns the cache spec describing per-block storage.

        Args:
            vllm_config: Global vLLM config.

        Returns:
            A ``KVCacheSpec`` subclass instance.
        """
        ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        """Returns ``SkyLightAttentionBackend``."""
        return SkyLightAttentionBackend

    def forward(self) -> None:
        """No-op — the cache has no forward computation."""
        pass


class SkyLightIndexerPQCache(SkyLightIndexerCache):
    """PQ-specific indexer cache.

    Returns ``PQCacheSpec`` so that vLLM allocates appropriately-sized blocks
    for codebook storage.

    Attributes:
        head_dim: Key dimension per token.
        dtype: Storage dtype (bf16/fp16).
        pq_bits: Bits per codebook index.
        group_factor: Number of PQ sub-vectors.
    """

    def __init__(
        self,
        head_dim: int,
        dtype: torch.dtype,
        pq_bits: int,
        group_factor: int,
        prefix: str,
        cache_config: CacheConfig,
    ) -> None:
        super().__init__(prefix, cache_config)
        self.head_dim: int = head_dim
        self.dtype: torch.dtype = dtype
        self.pq_bits: int = pq_bits
        self.group_factor: int = group_factor

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        """Returns ``PQCacheSpec`` for PQ codebook block allocation.

        Args:
            vllm_config: Global vLLM config.

        Returns:
            ``PQCacheSpec`` instance with PQ-specific parameters.
        """
        return PQCacheSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            dtype=self.dtype,
            pq_bits=self.pq_bits,
            group_factor=self.group_factor,
        )
