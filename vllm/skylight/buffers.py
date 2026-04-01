# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared output buffers for SkyLight sparse index selection."""

from dataclasses import dataclass

import torch


@dataclass
class SparseIndexBuffers:
    """Shared output buffers for sparse index selection, allocated once at model
    init and reused across all layers.

    Only ``lens[b, h, q, 0]`` entries are valid in the corresponding rows of
    ``indices`` and ``weights``.

    Attributes:
        indices: Token indices selected by the indexer.
            Shape ``(batch, query_heads, queries, max_tokens)``, dtype int32.
        weights: Scores / weights for the selected tokens.
            Shape ``(batch, query_heads, queries, max_tokens)``, dtype float32.
        lens: Number of valid entries per row.
            Shape ``(batch, query_heads, queries, 1)``, dtype int32.
    """

    indices: torch.Tensor
    weights: torch.Tensor
    lens: torch.Tensor
