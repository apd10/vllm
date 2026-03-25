# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from pydantic import ValidationError

from vllm.config import VllmConfig
from vllm.config.skylight import SkylightConfig


def _full_pq_kwargs(**overrides):
    """Return a complete set of pqcache kwargs."""
    defaults = dict(
        use_sparse_attention=True,
        sparse_attention_type="pqcache",
        pq_group_factor=4,
        pq_bits=8,
        pq_kmeans_iter=20,
        pq_init_offset=0,
        pq_metric="l2",
    )
    defaults.update(overrides)
    return defaults


def _full_pqs_kwargs(**overrides):
    """Return a complete set of pqcache_sample kwargs."""
    defaults = dict(
        use_sparse_attention=True,
        sparse_attention_type="pqcache_sample",
        pqs_group_factor=4,
        pqs_bits=8,
        pqs_kmeans_iter=20,
        pqs_init_offset=0,
        pqs_metric="l2",
        pqs_base_sample_size=0.25,
    )
    defaults.update(overrides)
    return defaults


# --- 1. Default construction ---

def test_default_construction():
    cfg = SkylightConfig()
    assert cfg.use_sparse_attention is False
    assert cfg.sparse_attention_type == "none"
    assert cfg.pq_group_factor is None
    assert cfg.pq_bits is None
    assert cfg.pq_kmeans_iter is None
    assert cfg.pq_init_offset is None
    assert cfg.pq_metric is None
    assert cfg.pqs_group_factor is None
    assert cfg.pqs_bits is None
    assert cfg.pqs_kmeans_iter is None
    assert cfg.pqs_init_offset is None
    assert cfg.pqs_metric is None
    assert cfg.pqs_base_sample_size is None


def test_vllm_config_default_is_none():
    cfg = VllmConfig()
    assert cfg.skylight_config is None


# --- 2. Dry-run mode ---

def test_dry_run_mode():
    cfg = SkylightConfig(use_sparse_attention=True, sparse_attention_type="none")
    assert cfg.use_sparse_attention is True
    assert cfg.sparse_attention_type == "none"


# --- 3. PQCache valid ---

def test_pqcache_valid():
    cfg = SkylightConfig(**_full_pq_kwargs())
    assert cfg.sparse_attention_type == "pqcache"
    assert cfg.pq_group_factor == 4
    assert cfg.pq_bits == 8


# --- 4. PQCache-Sample valid ---

def test_pqcache_sample_valid():
    cfg = SkylightConfig(**_full_pqs_kwargs())
    assert cfg.sparse_attention_type == "pqcache_sample"
    assert cfg.pqs_base_sample_size == 0.25


# --- 5. PQCache missing fields ---

@pytest.mark.parametrize("missing_field", [
    "pq_group_factor",
    "pq_bits",
    "pq_kmeans_iter",
    "pq_init_offset",
    "pq_metric",
])
def test_pqcache_missing_fields_raises(missing_field):
    kwargs = _full_pq_kwargs()
    kwargs[missing_field] = None
    with pytest.raises(ValidationError, match="pq_\\* fields are required"):
        SkylightConfig(**kwargs)


# --- 6. PQCache-Sample missing fields ---

@pytest.mark.parametrize("missing_field", [
    "pqs_group_factor",
    "pqs_bits",
    "pqs_kmeans_iter",
    "pqs_init_offset",
    "pqs_metric",
    "pqs_base_sample_size",
])
def test_pqcache_sample_missing_fields_raises(missing_field):
    kwargs = _full_pqs_kwargs()
    kwargs[missing_field] = None
    with pytest.raises(ValidationError, match="pqs_\\* fields are required"):
        SkylightConfig(**kwargs)


# --- 7. Disabled with type set ---

def test_disabled_with_type_set_raises():
    with pytest.raises(
        ValidationError,
        match="Cannot select a sparse attention type",
    ):
        SkylightConfig(
            use_sparse_attention=False,
            sparse_attention_type="pqcache",
        )


# --- 8. PQ fields set for wrong type ---

def test_pq_fields_set_for_wrong_type_raises():
    kwargs = _full_pqs_kwargs()
    kwargs["pq_group_factor"] = 4
    with pytest.raises(
        ValidationError,
        match="pq_\\* fields should only be set",
    ):
        SkylightConfig(**kwargs)


# --- 9. PQS fields set for wrong type ---

def test_pqs_fields_set_for_wrong_type_raises():
    kwargs = _full_pq_kwargs()
    kwargs["pqs_group_factor"] = 4
    with pytest.raises(
        ValidationError,
        match="pqs_\\* fields should only be set",
    ):
        SkylightConfig(**kwargs)


# --- 10. Dry-run with subfields ---

def test_dry_run_with_subfields_raises():
    with pytest.raises(
        ValidationError,
        match="No pq_\\*/pqs_\\* fields should be set in dry-run mode",
    ):
        SkylightConfig(
            use_sparse_attention=True,
            sparse_attention_type="none",
            pq_bits=8,
        )


# --- 11. Invalid type ---

def test_invalid_type_raises():
    with pytest.raises(ValidationError):
        SkylightConfig(
            use_sparse_attention=True,
            sparse_attention_type="unknown",
        )


# --- 12. Extra fields forbidden ---

def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        SkylightConfig(nonexistent_field=42)


# --- 13. Sample size float fraction ---

def test_sample_size_float_fraction():
    cfg = SkylightConfig(**_full_pqs_kwargs(pqs_base_sample_size=0.5))
    assert cfg.pqs_base_sample_size == 0.5


# --- 14. Sample size int absolute ---

def test_sample_size_int_absolute():
    cfg = SkylightConfig(**_full_pqs_kwargs(pqs_base_sample_size=128))
    assert cfg.pqs_base_sample_size == 128


# --- 15. Sample size float out of range ---

def test_sample_size_float_out_of_range_raises():
    with pytest.raises(ValidationError, match="pqs_base_sample_size as float"):
        SkylightConfig(**_full_pqs_kwargs(pqs_base_sample_size=1.5))


# --- 16. Sample size int zero ---

def test_sample_size_int_zero_raises():
    with pytest.raises(ValidationError, match="pqs_base_sample_size as int"):
        SkylightConfig(**_full_pqs_kwargs(pqs_base_sample_size=0))


# --- 17. Hash deterministic ---

def test_compute_hash_deterministic():
    cfg1 = SkylightConfig(**_full_pq_kwargs())
    cfg2 = SkylightConfig(**_full_pq_kwargs())
    assert cfg1.compute_hash() == cfg2.compute_hash()

    cfg_different = SkylightConfig(**_full_pq_kwargs(pq_bits=4))
    assert cfg1.compute_hash() != cfg_different.compute_hash()


# --- 18. Hash changes with fields ---

def test_compute_hash_changes_with_fields():
    base = SkylightConfig(**_full_pq_kwargs())
    base_hash = base.compute_hash()

    for field_name, alt_value in [
        ("pq_group_factor", 8),
        ("pq_bits", 4),
        ("pq_kmeans_iter", 10),
        ("pq_init_offset", 1),
        ("pq_metric", "cosine"),
    ]:
        altered = SkylightConfig(**_full_pq_kwargs(**{field_name: alt_value}))
        assert altered.compute_hash() != base_hash, (
            f"Hash should change when {field_name} changes"
        )


# --- 19. VllmConfig integration ---

def test_vllm_config_integration():
    cfg_none = VllmConfig()
    assert cfg_none.skylight_config is None
    hash_without = cfg_none.compute_hash()

    skylight = SkylightConfig()
    cfg_with = VllmConfig(skylight_config=skylight)
    assert cfg_with.skylight_config is skylight
    hash_with_default = cfg_with.compute_hash()
    assert hash_with_default != hash_without

    cfg_active = VllmConfig(
        skylight_config=SkylightConfig(
            use_sparse_attention=True,
            sparse_attention_type="none",
        )
    )
    assert cfg_active.compute_hash() != hash_without
    assert cfg_active.compute_hash() != hash_with_default


# --- 20. Backend mismatch raises ---

def test_backend_mismatch_raises():
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    non_skylight_backends = [
        b for b in AttentionBackendEnum if b.value != "SKYLIGHT"
    ]
    if not non_skylight_backends:
        pytest.skip("No non-SKYLIGHT backends available to test against")

    backend = non_skylight_backends[0]
    with pytest.raises(ValueError, match="Skylight sparse attention requires"):
        VllmConfig(
            skylight_config=SkylightConfig(
                use_sparse_attention=True,
                sparse_attention_type="none",
            ),
            attention_config={"backend": backend},
        )


# --- 21. JSON round-trip ---

def test_json_round_trip():
    json_dict = dict(
        use_sparse_attention=True,
        sparse_attention_type="pqcache",
        pq_group_factor=4,
        pq_bits=8,
        pq_kmeans_iter=20,
        pq_init_offset=0,
        pq_metric="l2",
    )
    from_dict = SkylightConfig(**json_dict)
    from_kwargs = SkylightConfig(
        use_sparse_attention=True,
        sparse_attention_type="pqcache",
        pq_group_factor=4,
        pq_bits=8,
        pq_kmeans_iter=20,
        pq_init_offset=0,
        pq_metric="l2",
    )
    assert from_dict.compute_hash() == from_kwargs.compute_hash()
    assert from_dict.pq_group_factor == from_kwargs.pq_group_factor
    assert from_dict.pq_bits == from_kwargs.pq_bits
