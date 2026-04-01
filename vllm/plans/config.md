# Plan: Add `SkylightConfig` to vLLM

## Goal

Add a new config concern `SkylightConfig` in `vllm/config/skylight.py` that controls sparse attention behaviour, and wire it into `VllmConfig` following existing conventions.

---

## 1. Define `SkylightConfig` dataclass

**File:** `vllm/config/skylight.py`

```python
@config
class SkylightConfig:
    use_sparse_attention: bool = False
    sparse_attention_type: Literal["none", "pqcache", "pqcache_sample"] = "none"

    # --- PQCache options ---
    pq_group_factor: int | None = None
    pq_bits: int | None = None
    pq_kmeans_iter: int | None = None
    pq_init_offset: int | None = None
    pq_metric: str | None = None

    # --- PQCache-Sample options ---
    pqs_group_factor: int | None = None
    pqs_bits: int | None = None
    pqs_kmeans_iter: int | None = None
    pqs_init_offset: int | None = None
    pqs_metric: str | None = None
    pqs_base_sample_size: float | int | None = None

    def compute_hash(self) -> str: ...
```

### Design decisions (resolved)

1. **Flat layout with validation.** PQCache/PQS fields live directly on
   `SkylightConfig` with `None` defaults. A `model_validator` enforces that the
   correct group is populated for the selected `sparse_attention_type`. We can
   refactor to nested sub-dataclasses later if the option count grows.

2. **`use_sparse_attention` and `sparse_attention_type` are both needed.** They
   serve distinct purposes:

   | `use_sparse_attention` | `sparse_attention_type` | Behaviour |
   |------------------------|------------------------|-----------|
   | `False` | `"none"` | Default. Sparse attention code is **not invoked** at all. Standard vLLM attention path. |
   | `True` | `"none"` | **Dry-run mode.** The new Skylight code path is invoked, but it computes full (dense) attention. Useful for testing the code path without changing results. |
   | `True` | `"pqcache"` | PQCache sparse attention is active. All `pq_*` fields must be set. |
   | `True` | `"pqcache_sample"` | PQCache-Sample sparse attention is active. All `pqs_*` fields must be set. |
   | `False` | `"pqcache"` / `"pqcache_sample"` | **Invalid.** Raises `ValidationError`. |

3. **`compute_hash` participation.** Sparse attention changes the computation
   graph, so all fields should be hashed using the standard
   `get_hash_factors` / `hash_factors` utilities (matching `AttentionConfig`).

4. **Default values for PQCache/PQS fields.** Kept as `None`. Users must
   explicitly provide all sub-fields when selecting a sparse attention type.
   We can add sensible defaults later once the runtime interface stabilises.

5. **`pqs_base_sample_size` is `float | int`.** The type determines
   interpretation:
   - `float` — fraction of tokens to sample (e.g. `0.25` = 25%).
   - `int` — absolute number of tokens to sample (e.g. `128`).

   A `field_validator` will enforce that float values are in `(0.0, 1.0]` and
   int values are `>= 1`.

---

## 2. Wire into `VllmConfig`

**File:** `vllm/config/vllm.py`

| Step | Change |
|------|--------|
| Import | Add `from .skylight import SkylightConfig` to the import block (line ~28-48). |
| Field | Add `skylight_config: SkylightConfig = Field(default_factory=SkylightConfig)` alongside the other always-present configs (~line 298). Always-present since `use_sparse_attention=False` is a clean default and avoids `None` checks everywhere. |
| `compute_hash` | Add `vllm_factors.append(self.skylight_config.compute_hash())` in `compute_hash()` (~line 406). |
| `__post_init__` | Add attention-backend guard (see section 5.2). |
| `__str__` | Add `skylight_config={self.skylight_config!r}` to the string representation (~line 1654). |

---

## 3. Wire into `__init__.py`

**File:** `vllm/config/__init__.py`

- Add `from vllm.config.skylight import SkylightConfig` to the imports.
- Add `"SkylightConfig"` to `__all__`.

---

## 4. Wire into `EngineArgs`

**File:** `vllm/engine/arg_utils.py`

| Step | Change |
|------|--------|
| Import | Add `SkylightConfig` to the config imports. |
| Field | Add `skylight_config: SkylightConfig = get_field(VllmConfig, "skylight_config")`. |
| `__post_init__` | Add dict-to-dataclass coercion: `if isinstance(self.skylight_config, dict): self.skylight_config = SkylightConfig(**self.skylight_config)`. |
| `add_cli_args` | Add `--skylight-config` argument (JSON blob, same pattern as `--kernel-config`). |
| `create_engine_config` | Pass `skylight_config=self.skylight_config` into the `VllmConfig(...)` constructor. |

---

## 5. Validation

### 5.1 Internal validation (inside `SkylightConfig`)

Add a `@model_validator(mode="after")` (Pydantic v2 style) that enforces **all**
of the following:

| Rule | Condition | Error |
|------|-----------|-------|
| Kill-switch consistency | `use_sparse_attention is False` and `sparse_attention_type != "none"` | `"Cannot select a sparse attention type when use_sparse_attention is False"` |
| PQCache completeness | `sparse_attention_type == "pqcache"` and any `pq_*` field is `None` | `"All pq_* fields are required when sparse_attention_type is 'pqcache'"` |
| PQS completeness | `sparse_attention_type == "pqcache_sample"` and any `pqs_*` field is `None` | `"All pqs_* fields are required when sparse_attention_type is 'pqcache_sample'"` |
| PQ fields unused | `sparse_attention_type != "pqcache"` and any `pq_*` field is not `None` | `"pq_* fields should only be set when sparse_attention_type is 'pqcache'"` |
| PQS fields unused | `sparse_attention_type != "pqcache_sample"` and any `pqs_*` field is not `None` | `"pqs_* fields should only be set when sparse_attention_type is 'pqcache_sample'"` |
| Dry-run clean | `use_sparse_attention is True` and `sparse_attention_type == "none"` and any sub-field is not `None` | `"No pq_*/pqs_* fields should be set in dry-run mode (sparse_attention_type='none')"` |

Add a `@field_validator("pqs_base_sample_size")` that enforces:
- If `float`: must be in `(0.0, 1.0]` (fraction of tokens).
- If `int`: must be `>= 1` (absolute token count).

### 5.2 Cross-config validation (inside `VllmConfig.__post_init__`)

Add a guard in `__post_init__` that enforces:

```
if self.skylight_config.use_sparse_attention:
    # A dedicated Skylight attention backend will be added in a future
    # change. For now, gate on that backend name so we fail early if
    # someone enables sparse attention with an incompatible backend.
    SKYLIGHT_BACKEND = "SKYLIGHT"  # placeholder, updated when backend lands
    if (self.attention_config.backend is not None
            and self.attention_config.backend != SKYLIGHT_BACKEND):
        raise ValueError(
            f"Skylight sparse attention requires the '{SKYLIGHT_BACKEND}' "
            f"attention backend, but got '{self.attention_config.backend}'."
        )
```

This is a **forward-looking guard**. When the Skylight attention backend is
added in a later change, we update `SKYLIGHT_BACKEND` to its real enum value.
Until that backend exists, enabling `use_sparse_attention=True` without setting
the backend will pass (since `backend=None` means auto-select), and the runtime
will resolve the backend later.

---

## 6. Unit tests

**File:** `tests/config/test_skylight_config.py`

Tests go in the existing `tests/config/` directory alongside `test_config_utils.py`
and `test_config_generation.py`.

### Test cases

| # | Test | What it verifies |
|---|------|------------------|
| 1 | `test_default_construction` | `SkylightConfig()` produces valid defaults (`use_sparse_attention=False`, `sparse_attention_type="none"`, all sub-fields `None`). |
| 2 | `test_dry_run_mode` | `SkylightConfig(use_sparse_attention=True, sparse_attention_type="none")` succeeds — the dry-run configuration is valid. |
| 3 | `test_pqcache_valid` | Constructing with `use_sparse_attention=True`, `sparse_attention_type="pqcache"` and all `pq_*` fields set succeeds. |
| 4 | `test_pqcache_sample_valid` | Same for `"pqcache_sample"` with all `pqs_*` fields. |
| 5 | `test_pqcache_missing_fields_raises` | `sparse_attention_type="pqcache"` with any `pq_*` field left as `None` raises `ValidationError`. |
| 6 | `test_pqcache_sample_missing_fields_raises` | Same for `"pqcache_sample"`. |
| 7 | `test_disabled_with_type_set_raises` | `use_sparse_attention=False` + `sparse_attention_type="pqcache"` raises `ValidationError`. |
| 8 | `test_pq_fields_set_for_wrong_type_raises` | `sparse_attention_type="pqcache_sample"` with `pq_*` fields set raises (wrong group). |
| 9 | `test_pqs_fields_set_for_wrong_type_raises` | `sparse_attention_type="pqcache"` with `pqs_*` fields set raises (wrong group). |
| 10 | `test_dry_run_with_subfields_raises` | `use_sparse_attention=True`, `sparse_attention_type="none"` with any `pq_*`/`pqs_*` field set raises. |
| 11 | `test_invalid_type_raises` | `sparse_attention_type="unknown"` raises `ValidationError`. |
| 12 | `test_extra_fields_forbidden` | Passing an unknown kwarg raises (enforced by `@config` decorator's `extra="forbid"`). |
| 13 | `test_sample_size_float_fraction` | `pqs_base_sample_size=0.25` (valid float fraction) accepted. |
| 14 | `test_sample_size_int_absolute` | `pqs_base_sample_size=128` (valid int count) accepted. |
| 15 | `test_sample_size_float_out_of_range_raises` | `pqs_base_sample_size=1.5` raises (float must be in `(0, 1]`). |
| 16 | `test_sample_size_int_zero_raises` | `pqs_base_sample_size=0` raises (int must be `>= 1`). |
| 17 | `test_compute_hash_deterministic` | Same inputs produce same hash; different inputs produce different hashes. |
| 18 | `test_compute_hash_changes_with_fields` | Changing any single field changes the hash. |
| 19 | `test_vllm_config_integration` | `VllmConfig(skylight_config=SkylightConfig(...))` round-trips correctly; `VllmConfig.compute_hash` includes skylight contribution. |
| 20 | `test_backend_mismatch_raises` | `use_sparse_attention=True` with an incompatible attention backend raises `ValueError` in `VllmConfig.__post_init__`. |
| 21 | `test_json_round_trip` | Constructing from a dict (simulating CLI `--skylight-config '{...}'`) produces the same object as keyword construction. |

---

## 7. Files changed (summary)

| File | Action |
|------|--------|
| `vllm/config/skylight.py` | **New** — `SkylightConfig` class |
| `vllm/config/__init__.py` | Add import + `__all__` entry |
| `vllm/config/vllm.py` | Add field, import, hash, `__str__`, backend guard in `__post_init__` |
| `vllm/engine/arg_utils.py` | Add CLI arg + wiring |
| `tests/config/test_skylight_config.py` | **New** — unit tests |
