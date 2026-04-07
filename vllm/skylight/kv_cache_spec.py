"""KV cache spec definitions for Skylight indexer caches."""

from dataclasses import dataclass

from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import KVCacheSpec


@dataclass(frozen=True)
class SkylightSparseAttentionPQCacheIndexerSpec(KVCacheSpec):
    """KV cache spec for the Skylight PQCache indexer.

    Describes a per-layer product-quantized index that stores compact codes for
    each token. The index is used at decode time for top-k token selection
    before reading the full KV cache.

    Attributes:
        pq_bits: Number of bits per PQ sub-quantizer.
        pq_group_factor: Number of PQ sub-quantizer groups.
        page_size_padded: Optional override for page size, used in hybrid
            models to align with attention page size when block_size inflation
            alone cannot produce an exact match.
    """

    pq_bits: int
    pq_group_factor: int
    page_size_padded: int | None = None

    @property
    def page_size_bytes(self) -> int:
        real_size: int = self.real_page_size_bytes
        if self.page_size_padded is not None:
            assert self.page_size_padded >= real_size
            return self.page_size_padded
        return real_size

    @property
    def real_page_size_bytes(self) -> int:
        bytes_per_token: int = (self.pq_group_factor * self.pq_bits) // 8
        return self.block_size * bytes_per_token

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        """Return the maximum memory used by this PQ indexer cache."""
        max_model_len: int = vllm_config.model_config.max_model_len
        dcp_world_size: int = vllm_config.parallel_config.decode_context_parallel_size
        pcp_world_size: int = vllm_config.parallel_config.prefill_context_parallel_size
        if dcp_world_size * pcp_world_size > 1:
            max_model_len = cdiv(max_model_len, dcp_world_size * pcp_world_size)
        return cdiv(max_model_len, self.block_size) * self.page_size_bytes
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the skylight project

"""PQCacheSpec — KVCacheSpec subclass for PQ indexer cache blocks."""

from dataclasses import dataclass
from math import ceil

import torch

from vllm.v1.kv_cache_interface import KVCacheSpec


@dataclass(frozen=True)
class PQCacheSpec(KVCacheSpec):
    """Describes per-block memory layout for the PQ indexer cache.

    The indexer cache stores codebook indices (int8) produced by product
    quantisation.  ``page_size_bytes`` tells vLLM how many bytes a single
    block of ``block_size`` tokens requires.

    Attributes:
        block_size: Number of tokens per block (inherited from KVCacheSpec).
        num_kv_heads: Number of KV heads for the indexer cache (can be > 1).
        dtype: Storage dtype — ``torch.int8`` (codebook indices).
        pq_bits: Bits per codebook index (codebook size = 2^pq_bits).
        group_factor: Number of PQ sub-vectors per head.
    """

    num_kv_heads: int
    dtype: torch.dtype = torch.int8
    pq_bits: int = 8
    group_factor: int = 1

    @property
    def page_size_bytes(self) -> int:
        """Number of bytes per block.

        Formula: ``ceil(pq_bits / 8) * group_factor * num_kv_heads * block_size``.
        """
        return ceil(self.pq_bits / 8) * self.group_factor * self.num_kv_heads * self.block_size
