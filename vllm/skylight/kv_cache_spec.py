# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
