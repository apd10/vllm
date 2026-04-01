# Plan: Add `SkylightSparseAttentionPQCacheIndexerSpec` KVCacheSpec

## Goal

Add a new `KVCacheSpec` subclass for the Skylight PQCache indexer. This spec
describes an auxiliary per-layer index structure (product-quantised codes) that
lives alongside the regular KV cache and is managed by vLLM's block allocator.

---

## 1. Context: What this spec represents

The PQCache indexer stores product-quantised (PQ) codes for each token's key
vectors. These codes are used at decode time to quickly identify which tokens
are relevant (top-k selection) before reading the full KV cache.

Each token's index entry is `(pq_group_factor * pq_bits) / 8` bytes. The dtype
is fixed at `int8` (the PQ codes are packed bit fields, not real int8 values,
but we store them as int8 for allocation purposes).

This spec does **not** inherit from `AttentionSpec` because it does not describe
K/V head storage — it describes an auxiliary index. It inherits directly from
`KVCacheSpec`.

---

## 2. Define the dataclass

**File:** `vllm/v1/kv_cache_interface.py`

```python
@dataclass(frozen=True)
class SkylightSparseAttentionPQCacheIndexerSpec(KVCacheSpec):
    """KV cache spec for the Skylight PQCache indexer.

    Describes a per-layer product-quantised index that stores compact
    codes for each token. The index is used at decode time for top-k
    token selection before reading the full KV cache.

    Attributes:
        pq_bits: Number of bits per PQ sub-quantiser.
        pq_group_factor: Number of PQ sub-quantiser groups.
        page_size_padded: Optional override for page size, used in hybrid
            models to align with attention page size when block_size
            inflation alone cannot produce an exact match.
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
        max_model_len: int = vllm_config.model_config.max_model_len
        dcp_world_size: int = (
            vllm_config.parallel_config.decode_context_parallel_size
        )
        pcp_world_size: int = (
            vllm_config.parallel_config.prefill_context_parallel_size
        )
        if dcp_world_size * pcp_world_size > 1:
            max_model_len = cdiv(
                max_model_len, dcp_world_size * pcp_world_size
            )
        return cdiv(max_model_len, self.block_size) * self.page_size_bytes
```

### Design decisions

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | Inherit from `KVCacheSpec` directly, not `AttentionSpec` | This is an index structure, not a KV head store. It has no `num_kv_heads`, `head_size`, or `dtype` in the `AttentionSpec` sense. |
| 2 | Fixed `int8` dtype not stored as a field | The PQ codes are always stored as packed bytes; there is no user-facing dtype choice. The `page_size_bytes` formula computes the byte count directly. |
| 3 | `max_memory_usage_bytes` mirrors `FullAttentionSpec` | The indexer needs one entry per token (up to `max_model_len`), same as full attention. DCP/PCP partitioning applies identically. |
| 4 | `page_size_padded` field included | Follows the `MambaSpec` pattern. Used only as a fallback for hybrid models when block_size inflation cannot produce an exact page size match (see section 5). |
| 5 | Separate `real_page_size_bytes` property | Follows the `AttentionSpec` / `MambaSpec` pattern — `page_size_bytes` returns the padded size if set, `real_page_size_bytes` always returns the true data size. Useful for the reshape step in the model runner. |
| 6 | Default `merge()` from `KVCacheSpec` is sufficient | All indexer layers with the same `pq_bits` and `pq_group_factor` are identical; the base `merge()` asserts equality and deep-copies, which is correct. |

---

## 3. Register in `UniformTypeKVCacheSpecs.is_uniform_type`

**File:** `vllm/v1/kv_cache_interface.py`, inside `UniformTypeKVCacheSpecs.is_uniform_type`

### Background: What `UniformTypeKVCacheSpecs` does

`get_kv_cache_groups` in `kv_cache_utils.py` decides how to group layers for
block allocation. There are three tiers:

1. **Uniform spec** — all layers have *identical* specs (same class, same
   params). Single group, simplest path.
2. **Uniform type** (`UniformTypeKVCacheSpecs`) — all layers have the same
   *type* (e.g., all `FullAttentionSpec`) but potentially different parameters
   (e.g., different `num_kv_heads`). All layers still need the same number of
   token slots, so they share one block table. The composite
   `page_size_bytes` = sum of all individual specs' page sizes.
3. **Uniform page size** — different types (e.g., full + sliding window).
   `unify_kv_cache_spec_page_size` inflates smaller specs' `block_size` so all
   groups have equal physical page size. Multiple groups, each with a separate
   block table.

We add a new `elif` branch after the `MambaSpec` check (around line 426) for
tier 2 handling:

```python
elif isinstance(one_spec, SkylightSparseAttentionPQCacheIndexerSpec):
    return all(
        isinstance(spec, SkylightSparseAttentionPQCacheIndexerSpec)
        and spec.pq_bits == one_spec.pq_bits
        and spec.pq_group_factor == one_spec.pq_group_factor
        for spec in kv_cache_specs.values()
    )
```

Uniformity requires all indexer layers to have matching `pq_bits` and
`pq_group_factor` (otherwise they have different page sizes and cannot share a
block pool).

---

## 4. KV cache manager

**File:** `vllm/v1/core/single_type_kv_cache_manager.py`

### 4.1 Reuse `FullAttentionManager`

The PQCache indexer allocates blocks identically to full attention — one block
per `block_size` tokens, no sliding window, no chunking, standard prefix
caching. `FullAttentionManager` already handles this.

No new manager class is needed.

### 4.2 Register in `spec_manager_map`

Add entry:

```python
spec_manager_map: dict[type[KVCacheSpec], type[SingleTypeKVCacheManager]] = {
    ...
    SkylightSparseAttentionPQCacheIndexerSpec: FullAttentionManager,
}
```

### 4.3 Relax assertion in `FullAttentionManager.find_longest_cache_hit`

`FullAttentionManager.find_longest_cache_hit` asserts that the spec is
`FullAttentionSpec | ChunkedLocalAttentionSpec`. Relax this to also accept
`SkylightSparseAttentionPQCacheIndexerSpec`. The logic in `FullAttentionManager`
is type-agnostic; the assertion is a safety check that simply needs updating.

---

## 5. Hybrid models: Skylight + Mamba / SlidingWindow / other types

### 5.1 How hybrid models will look

A Skylight model will have two types of modules per transformer layer:
- A standard attention module returning `FullAttentionSpec` (actual K/V cache)
- A separate indexer module returning `SkylightSparseAttentionPQCacheIndexerSpec`
  (PQ codes)

This follows the same pattern as DeepSeek's `Indexer` class, which is a
separate `AttentionLayerBase` module with its own `get_kv_cache_spec()`.

When mixed with other layer types (Mamba, SlidingWindow), the model has 3+
different spec types.

### 5.2 Why page sizes must match across groups

In vLLM's hybrid model path (`_get_kv_cache_groups_uniform_page_size`), layers
from different groups **share the same physical tensor**. For example, with
groups `(full.0, full.1)` and `(indexer.0, indexer.1)`:
- Tensor 0 is shared by `full.0` and `indexer.0`
- Tensor 1 is shared by `full.1` and `indexer.1`

Since block ID N maps to the same byte offset in the shared tensor regardless
of which group is accessing it, all groups must report the same
`page_size_bytes`.

All groups also share a single `BlockPool`. Each group has its own block table
and independently allocates block IDs from the pool. Block IDs never overlap
between groups — a block can only be allocated to one group at a time.

### 5.3 How page size alignment works: block_size inflation

The primary mechanism is **block_size inflation** via
`unify_kv_cache_spec_page_size`. It inflates the smaller spec's `block_size`
so that its `page_size_bytes` matches the larger spec's page size. No memory
is wasted — the indexer block simply covers more tokens.

**Worked example** (Llama-style, bf16, 8 KV heads, 128 head dim):

```
memory_per_token_fw  = 2 * 8 * 128 * 2 = 4096 bytes/token  (K+V)
page_size_fw         = 16 * 4096       = 65536 bytes

memory_per_token_pq  = (8 * 4) / 8     = 4 bytes/token
page_size_pq_initial = 16 * 4          = 64 bytes

ratio                = 65536 / 64      = 1024
block_size_pq        = 16 * 1024       = 16384
page_size_pq_final   = 16384 * 4       = 65536 bytes  ← exact match
```

This is memory-efficient. For a request with 1000 tokens:

| Group | block_size | Blocks used | Memory consumed | Useful data |
|-------|-----------|-------------|-----------------|-------------|
| Attention | 16 | ceil(1000/16) = 63 | 63 × 65536 = 4.0 MB | 4.0 MB |
| PQ indexer | 16384 | ceil(1000/16384) = 1 | 1 × 65536 = 64 KB | 1000 × 4 = 4 KB |
| **Total** | | **64 blocks** | | |

Compare this with a naive `page_size_padded` approach (no inflation):

| Group | block_size | Blocks used | Memory consumed | Useful data |
|-------|-----------|-------------|-----------------|-------------|
| Attention | 16 | 63 | 63 × 65536 = 4.0 MB | 4.0 MB |
| PQ indexer | 16 | 63 | 63 × 65536 = 4.0 MB | 1000 × 4 = 4 KB |
| **Total** | | **126 blocks** | | |

Block_size inflation uses **half** the blocks, doubling the number of
concurrent requests that can fit in the block pool.

### 5.4 Fallback: `page_size_padded` for non-divisible cases

`unify_kv_cache_spec_page_size` requires
`page_size_fw % page_size_pq == 0`. When this fails (e.g., `pq_bits=6,
pq_group_factor=8` → 6 bytes/token → `65536 % 96 ≠ 0`), the function raises
`NotImplementedError`.

For these cases we combine inflation + padding:
1. Inflate block_size as far as possible:
   `block_size_pq = block_size_fw * floor(memory_per_token_fw / memory_per_token_pq)`
2. Pad the remainder via `page_size_padded`:
   `page_size_padded = page_size_fw`

**Example:** `memory_per_token_pq = 6 bytes/token`
```
block_size_pq   = 16 * floor(4096 / 6) = 16 * 682 = 10912
page_size_pq    = 10912 * 6             = 65472 bytes
page_size_padded = 65536                             (pad 64 bytes)
```

Waste is only 64 bytes per block — negligible.

This `page_size_padded` logic will be set at model init time (similar to how
`HybridAttentionMambaModelConfig` sets `mamba_page_size_padded`), and is part of
the attention layer wiring (out of scope for this PR — see section 6).

### 5.5 Practical considerations

- **Prefix caching granularity:** With `block_size_pq = 16384`, a PQ cache hit
  requires all 16384 tokens in a block to match. This is a non-issue — the
  attention group does fine-grained prefix caching at `block_size=16`
  independently. The PQ index is auxiliary and is recomputed for partial blocks.

- **Common PQ parameter choices are divisible.** `pq_bits ∈ {4, 8}` and
  `pq_group_factor ∈ {2, 4, 8, 16}` produce `memory_per_token_pq ∈ {1, 2, 4,
  8, 16}` bytes/token, all of which are powers of 2 and divide any typical
  attention page size evenly. The `page_size_padded` fallback is rarely needed.

### 5.6 What we need to do now: nothing

The existing `unify_kv_cache_spec_page_size` handles the common case (divisible
page sizes) automatically by inflating `block_size` via `copy_with_new_block_size()`.
No code changes are needed in this PR for hybrid model support.

The `page_size_padded` field is included on the spec for forward compatibility.
It will be wired at model init time when the Skylight attention layer is added
in a future PR.

---

## 6. Wire up `get_kv_cache_spec()` in the attention layer

This is **not** part of the current change. The Skylight attention layer
(which will use this spec) will be added in a separate PR. For now, the spec
class exists and is tested in isolation.

When the Skylight attention layer is added, its indexer module's
`get_kv_cache_spec()` will read from `SkylightConfig` (already wired via CLI):

```python
def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
    skylight_config = vllm_config.skylight_config
    return SkylightSparseAttentionPQCacheIndexerSpec(
        block_size=vllm_config.cache_config.block_size,
        pq_bits=skylight_config.pq_bits,
        pq_group_factor=skylight_config.pq_group_factor,
    )
```

**Note on CLI wiring:** `pq_bits` and `pq_group_factor` are already defined in
`SkylightConfig` (`vllm/config/skylight.py`) and exposed via the
`--skylight-config` CLI argument. No additional CLI plumbing is needed — the
spec reads these values at construction time from the config that is already
wired.

---

## 7. Unit tests

**File:** `tests/v1/core/test_skylight_kv_cache_spec.py`

Tests follow the patterns in `tests/v1/core/test_kv_cache_utils.py`.

### 7.1 Spec construction and properties

| # | Test | What it verifies |
|---|------|------------------|
| 1 | `test_construction` | `SkylightSparseAttentionPQCacheIndexerSpec(block_size=16, pq_bits=8, pq_group_factor=4)` constructs without error. Fields are accessible. |
| 2 | `test_page_size_bytes` | `page_size_bytes == block_size * (pq_group_factor * pq_bits) // 8`. Test with several parameter combinations. |
| 3 | `test_page_size_bytes_parametrized` | Parametrize over `(pq_bits, pq_group_factor, block_size)` combinations: `(8,4,16)`, `(4,8,16)`, `(8,8,32)`, `(2,16,8)`. Verify expected byte counts. |
| 4 | `test_page_size_bytes_with_padding` | Construct with `page_size_padded=65536`. Verify `page_size_bytes` returns 65536 and `real_page_size_bytes` returns the unpadded value. |
| 5 | `test_max_memory_usage_bytes` | Mock `VllmConfig` with known `max_model_len`, `dcp_world_size=1`, `pcp_world_size=1`. Verify result equals `cdiv(max_model_len, block_size) * page_size_bytes`. |
| 6 | `test_max_memory_usage_bytes_with_dcp` | Same as above but with `dcp_world_size=2`. Verify `max_model_len` is halved. |
| 7 | `test_frozen_dataclass` | Attempting to mutate a field raises `FrozenInstanceError`. |
| 8 | `test_copy_with_new_block_size` | `copy_with_new_block_size(32)` returns new spec with `block_size=32`, other fields unchanged. |

### 7.2 Merge behaviour

| # | Test | What it verifies |
|---|------|------------------|
| 9 | `test_merge_identical_specs` | Merging a list of identical specs returns an equal spec. |
| 10 | `test_merge_different_specs_raises` | Merging specs with different `pq_bits` raises `AssertionError`. |

### 7.3 Uniform type checks

| # | Test | What it verifies |
|---|------|------------------|
| 11 | `test_is_uniform_type_same_params` | `is_uniform_type` returns `True` for dict of specs with same `pq_bits` and `pq_group_factor`. |
| 12 | `test_is_uniform_type_different_params` | `is_uniform_type` returns `False` when `pq_bits` differs across layers. |
| 13 | `test_is_uniform_type_mixed_with_full_attention` | `is_uniform_type` returns `False` for a mix of indexer specs and `FullAttentionSpec`. |

### 7.4 Manager integration

| # | Test | What it verifies |
|---|------|------------------|
| 14 | `test_spec_manager_map_lookup` | `spec_manager_map[SkylightSparseAttentionPQCacheIndexerSpec]` returns `FullAttentionManager`. |
| 15 | `test_get_manager_for_kv_cache_spec` | `get_manager_for_kv_cache_spec(spec, ...)` returns a `FullAttentionManager` instance. |

### 7.5 Hybrid page size alignment

| # | Test | What it verifies |
|---|------|------------------|
| 16 | `test_unify_page_size_with_full_attention_divisible` | Given a dict with `FullAttentionSpec` (page=65536) and `SkylightSparseAttentionPQCacheIndexerSpec` (page=64), `unify_kv_cache_spec_page_size` inflates indexer `block_size` so both report page=65536. |
| 17 | `test_copy_with_new_block_size_preserves_pq_params` | After `copy_with_new_block_size(16384)`, `pq_bits` and `pq_group_factor` are unchanged and `page_size_bytes` is correctly recomputed. |

---

## 8. Files changed (summary)

| File | Action |
|------|--------|
| `vllm/v1/kv_cache_interface.py` | Add `SkylightSparseAttentionPQCacheIndexerSpec` dataclass; add `elif` in `is_uniform_type`. |
| `vllm/v1/core/single_type_kv_cache_manager.py` | Add entry in `spec_manager_map`; relax assertion in `FullAttentionManager.find_longest_cache_hit`. |
| `tests/v1/core/test_skylight_kv_cache_spec.py` | **New** — unit tests (17 test cases). |

---

## 9. Out of scope (future PRs)

| Item | Why deferred |
|------|-------------|
| Skylight attention layer / indexer module returning this spec | Separate concern — the layer itself, its backend, and forward pass are a different PR. |
| Model init logic to set `page_size_padded` for non-divisible cases | Analogous to `HybridAttentionMambaModelConfig`, will be added with the attention layer. |
| Actual PQ index kernel integration | Runtime concern, not KV cache spec concern. |
