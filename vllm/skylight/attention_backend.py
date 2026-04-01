# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Lightweight vLLM AttentionBackend for SkyLight indexer caches.

Contains:
- ``SkyLightIndexerMetadata`` — per-batch metadata dataclass.
- ``SkyLightMetadataBuilder`` — builds metadata from CommonAttentionMetadata.
- ``SkyLightAttentionBackend`` — backend registered with the vLLM framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch

from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    AttentionMetadataBuilder,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.kv_cache_interface import KVCacheSpec


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

@dataclass
class SkyLightIndexerMetadata(AttentionMetadata):
    """Per-step metadata consumed by SkyLightIndexer backends.

    Thin wrapper around fields extracted from ``CommonAttentionMetadata``.

    Attributes:
        slot_mapping: Maps each token to its slot in the paged cache.
        block_table: Maps each request to its physical cache blocks.
        seq_lens: Total context length per request.
        query_start_loc: Cumulative query token offsets.
        num_actual_tokens: Total tokens in batch (excluding padding).
        num_decodes: Number of decode requests in the batch.
        num_decode_tokens: Total tokens belonging to decode requests.
        num_prefills: Number of prefill requests in the batch.
        num_prefill_tokens: Total tokens belonging to prefill requests.
    """

    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    query_start_loc: torch.Tensor
    num_actual_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int


# ---------------------------------------------------------------------------
# Metadata builder
# ---------------------------------------------------------------------------

class SkyLightMetadataBuilder(AttentionMetadataBuilder[SkyLightIndexerMetadata]):
    """Builds ``SkyLightIndexerMetadata`` from the framework-provided
    ``CommonAttentionMetadata`` each step.

    Attributes:
        kv_cache_spec: Spec for this cache group (from ``__init__``).
        layer_names: Layer names sharing this builder (from ``__init__``).
        vllm_config: Global config (from ``__init__``).
        device: Target device (from ``__init__``).
    """

    reorder_batch_threshold: ClassVar[int] = 1

    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> SkyLightIndexerMetadata:
        """Build indexer metadata from ``common_attn_metadata``."""
        raise NotImplementedError

# ---------------------------------------------------------------------------
# Attention backend
# ---------------------------------------------------------------------------

class SkyLightAttentionBackend(AttentionBackend):
    """Lightweight vLLM ``AttentionBackend`` for SkyLight indexer caches.

    This backend does **not** run attention forward — it only provides cache
    shaping and metadata building hooks so that vLLM allocates and manages
    paged cache blocks for the indexer.
    """

    @staticmethod
    def get_name() -> str:
        """Backend identifier used for grouping and selection."""
        return "SKYLIGHT_INDEXER"

    @staticmethod
    def get_builder_cls() -> type[SkyLightMetadataBuilder]:
        """Returns the metadata builder class."""
        return SkyLightMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type:
        """No attention impl — indexer caches do not run attention forward."""
        raise NotImplementedError(
            "SkyLightAttentionBackend does not provide an attention impl."
        )

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        """Cache shape for the indexer's paged KV cache."""
        return (num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(1)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return []
