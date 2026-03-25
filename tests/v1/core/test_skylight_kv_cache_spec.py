# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

import pytest
import torch

from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import unify_kv_cache_spec_page_size
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    get_manager_for_kv_cache_spec,
    spec_manager_map,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    SkylightSparseAttentionPQCacheIndexerSpec,
    UniformTypeKVCacheSpecs,
)

pytestmark = pytest.mark.cpu_test


def _make_pq_spec(
    block_size: int = 16,
    pq_bits: int = 8,
    pq_group_factor: int = 4,
    page_size_padded: int | None = None,
) -> SkylightSparseAttentionPQCacheIndexerSpec:
    return SkylightSparseAttentionPQCacheIndexerSpec(
        block_size=block_size,
        pq_bits=pq_bits,
        pq_group_factor=pq_group_factor,
        page_size_padded=page_size_padded,
    )


def _make_vllm_config(
    max_model_len: int = 2048,
    dcp_world_size: int = 1,
    pcp_world_size: int = 1,
) -> MagicMock:
    """Build a minimal mock VllmConfig for max_memory_usage_bytes tests."""
    cfg = MagicMock()
    cfg.model_config.max_model_len = max_model_len
    cfg.parallel_config.decode_context_parallel_size = dcp_world_size
    cfg.parallel_config.prefill_context_parallel_size = pcp_world_size
    return cfg


# ---------------------------------------------------------------------------
# 7.1  Spec construction and properties
# ---------------------------------------------------------------------------


def test_construction():
    spec = _make_pq_spec(block_size=16, pq_bits=8, pq_group_factor=4)
    assert spec.block_size == 16
    assert spec.pq_bits == 8
    assert spec.pq_group_factor == 4
    assert spec.page_size_padded is None


@pytest.mark.parametrize(
    ("pq_bits", "pq_group_factor", "block_size", "expected_page_size"),
    [
        (8, 4, 16, 16 * 4),
        (4, 8, 16, 16 * 4),
        (8, 8, 32, 32 * 8),
        (2, 16, 8, 8 * 4),
    ],
    ids=["8bit-4grp-bs16", "4bit-8grp-bs16", "8bit-8grp-bs32", "2bit-16grp-bs8"],
)
def test_page_size_bytes_parametrized(
    pq_bits: int,
    pq_group_factor: int,
    block_size: int,
    expected_page_size: int,
):
    spec = _make_pq_spec(
        block_size=block_size,
        pq_bits=pq_bits,
        pq_group_factor=pq_group_factor,
    )
    assert spec.page_size_bytes == expected_page_size
    assert spec.real_page_size_bytes == expected_page_size


def test_page_size_bytes_with_padding():
    spec = _make_pq_spec(
        block_size=16, pq_bits=8, pq_group_factor=4, page_size_padded=65536
    )
    assert spec.page_size_bytes == 65536
    assert spec.real_page_size_bytes == 16 * 4  # unpadded


def test_max_memory_usage_bytes():
    spec = _make_pq_spec(block_size=16, pq_bits=8, pq_group_factor=4)
    cfg = _make_vllm_config(max_model_len=2048)
    expected: int = cdiv(2048, 16) * spec.page_size_bytes
    assert spec.max_memory_usage_bytes(cfg) == expected


def test_max_memory_usage_bytes_with_dcp():
    spec = _make_pq_spec(block_size=16, pq_bits=8, pq_group_factor=4)
    cfg = _make_vllm_config(max_model_len=2048, dcp_world_size=2)
    effective_len: int = cdiv(2048, 2)
    expected: int = cdiv(effective_len, 16) * spec.page_size_bytes
    assert spec.max_memory_usage_bytes(cfg) == expected


def test_frozen_dataclass():
    spec = _make_pq_spec()
    with pytest.raises(FrozenInstanceError):
        spec.pq_bits = 16  # type: ignore[misc]


def test_copy_with_new_block_size():
    spec = _make_pq_spec(block_size=16, pq_bits=8, pq_group_factor=4)
    new_spec = spec.copy_with_new_block_size(32)
    assert new_spec.block_size == 32
    assert new_spec.pq_bits == 8
    assert new_spec.pq_group_factor == 4
    assert new_spec.page_size_bytes == 32 * 4


# ---------------------------------------------------------------------------
# 7.2  Merge behaviour
# ---------------------------------------------------------------------------


def test_merge_identical_specs():
    specs = [_make_pq_spec() for _ in range(3)]
    merged = SkylightSparseAttentionPQCacheIndexerSpec.merge(specs)
    assert merged == specs[0]


def test_merge_different_specs_raises():
    specs = [
        _make_pq_spec(pq_bits=8),
        _make_pq_spec(pq_bits=4),
    ]
    with pytest.raises(AssertionError):
        SkylightSparseAttentionPQCacheIndexerSpec.merge(specs)


# ---------------------------------------------------------------------------
# 7.3  Uniform type checks
# ---------------------------------------------------------------------------


def test_is_uniform_type_same_params():
    specs: dict[str, SkylightSparseAttentionPQCacheIndexerSpec] = {
        "layer0": _make_pq_spec(pq_bits=8, pq_group_factor=4),
        "layer1": _make_pq_spec(pq_bits=8, pq_group_factor=4),
    }
    assert UniformTypeKVCacheSpecs.is_uniform_type(specs)


def test_is_uniform_type_different_params():
    """Layers with different PQ parameters are still uniform-type (same class),
    matching the FullAttentionSpec pattern where different head counts still
    pass is_uniform_type."""
    specs: dict[str, SkylightSparseAttentionPQCacheIndexerSpec] = {
        "layer0": _make_pq_spec(pq_bits=8, pq_group_factor=4),
        "layer1": _make_pq_spec(pq_bits=4, pq_group_factor=8),
    }
    assert UniformTypeKVCacheSpecs.is_uniform_type(specs)


def test_is_uniform_type_mixed_with_full_attention():
    specs: dict[str, FullAttentionSpec | SkylightSparseAttentionPQCacheIndexerSpec] = {
        "layer0": _make_pq_spec(),
        "layer1": FullAttentionSpec(
            block_size=16,
            num_kv_heads=2,
            head_size=64,
            dtype=torch.float16,
        ),
    }
    assert not UniformTypeKVCacheSpecs.is_uniform_type(specs)


# ---------------------------------------------------------------------------
# 7.4  Manager integration
# ---------------------------------------------------------------------------


def test_spec_manager_map_lookup():
    assert (
        spec_manager_map[SkylightSparseAttentionPQCacheIndexerSpec]
        is FullAttentionManager
    )


def test_get_manager_for_kv_cache_spec():
    spec = _make_pq_spec()
    block_pool = MagicMock()
    block_pool.free_block_queue.__len__ = lambda _: 100
    manager = get_manager_for_kv_cache_spec(
        kv_cache_spec=spec,
        block_pool=block_pool,
        enable_caching=False,
        kv_cache_group_id=0,
    )
    assert isinstance(manager, FullAttentionManager)


# ---------------------------------------------------------------------------
# 7.5  Hybrid page size alignment
# ---------------------------------------------------------------------------


def test_unify_page_size_with_full_attention_divisible():
    """When FullAttentionSpec has a larger page, unify_kv_cache_spec_page_size
    should inflate the indexer block_size so both report the same page size."""
    full_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=32,
        head_size=128,
        dtype=torch.float16,
    )
    pq_spec = _make_pq_spec(block_size=16, pq_bits=8, pq_group_factor=4)

    assert full_spec.page_size_bytes != pq_spec.page_size_bytes

    specs: dict[str, FullAttentionSpec | SkylightSparseAttentionPQCacheIndexerSpec] = {
        "attn.0": full_spec,
        "pq.0": pq_spec,
    }
    unified = unify_kv_cache_spec_page_size(specs)
    page_sizes = {s.page_size_bytes for s in unified.values()}
    assert len(page_sizes) == 1, f"Expected uniform page size, got {page_sizes}"

    unified_pq = unified["pq.0"]
    unified_full = unified["attn.0"]
    assert isinstance(unified_pq, SkylightSparseAttentionPQCacheIndexerSpec)
    assert unified_pq.pq_bits == pq_spec.pq_bits
    assert unified_pq.pq_group_factor == pq_spec.pq_group_factor
    assert unified_pq.block_size > pq_spec.block_size
    assert unified_pq.page_size_bytes == unified_full.page_size_bytes


def test_copy_with_new_block_size_preserves_pq_params():
    spec = _make_pq_spec(block_size=16, pq_bits=8, pq_group_factor=4)
    inflated = spec.copy_with_new_block_size(16384)
    assert inflated.pq_bits == 8
    assert inflated.pq_group_factor == 4
    assert inflated.block_size == 16384
    assert inflated.page_size_bytes == 16384 * 4
