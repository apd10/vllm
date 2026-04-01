# SkyLight Indexer Plan

## Goal

Create a generic indexer framework (`SkyLightIndexer`) with a backend-pluggable
design. First backend: **PQCache** (from `sparse_attention_hub`).

---

## Class Hierarchy

```
SkyLightIndexer (nn.Module)                          # top-level module, lives in model code
  ├── cache: SkyLightIndexerCache                    # manages KV cache spec + vLLM AttentionBackend
  ├── backend: SkyLightIndexerBackend (ABC)          # pluggable compute strategy
  │     ├── SkyLightPQBackendRef                     # reference pytorch impl (sparse_attention_hub)
  │     └── SkyLightPQBackend                        # fast CUDA impl (future)
  └── sparse_index_buffers: SparseIndexBuffers       # shared output buffers (reused across layers)

SkyLightIndexerCache (nn.Module, AttentionLayerBase)  # base cache (no KV storage)
  └── SkyLightIndexerPQCache                          # returns PQCacheSpec, stores codebook in cache

SkyLightAttentionBackend (AttentionBackend)            # lightweight vLLM framework backend
  └── SkyLightMetadataBuilder (AttentionMetadataBuilder)  # thin pass-through builder
```

---

## File Location

```
vllm/skylight/
  __init__.py
  indexer.py               # SkyLightIndexer
  cache.py                 # SkyLightIndexerCache, SkyLightIndexerPQCache
  backend.py               # SkyLightIndexerBackend (ABC — compute backend)
  pq_backend_ref.py        # SkyLightPQBackendRef
  pq_backend.py            # SkyLightPQBackend (future)
  buffers.py               # SparseIndexBuffers
  attention_backend.py     # SkyLightAttentionBackend + SkyLightMetadataBuilder + SkyLightIndexerMetadata
  kv_cache_spec.py         # PQCacheSpec
```

---

## Config

No new config dataclass. Reuse `ResearchAttentionConfig` from sparse_attention_hub,
which contains `masker_configs: List[MaskerConfig]`. For PQCache, the relevant
masker config is `PQCacheConfig(TopKMaskerConfig)` with fields:
- `heavy_size` (topk count)
- `pq_group_factor`, `pq_bits`, `kmeans_iter`, `init_offset`, `metric`

Model-level dimensions (`n_heads`, `head_dim`, `rope_dim`) come from the
model config, not from the sparse attention config.

---

## 1. `SparseIndexBuffers` — `vllm/skylight/buffers.py`

Shared output buffers, allocated once at model init, reused across all layers.

### Fields

| Field | Type | Shape | Description |
|---|---|---|---|
| `indices` | `torch.Tensor` (int32) | `(batch, query_heads, queries, max_tokens)` | Token indices selected by the indexer |
| `weights` | `torch.Tensor` (float32) | `(batch, query_heads, queries, max_tokens)` | Scores/weights for the selected tokens |
| `lens` | `torch.Tensor` (int32) | `(batch, query_heads, queries, 1)` | Number of valid entries per row in `indices` and `weights` |

### Methods

None — pure data container. Allocated externally and passed around.

### Notes

- Only `lens[b, h, q, 0]` entries are valid in `indices[b, h, q, :]` and `weights[b, h, q, :]`.
- Allocated with `max_num_batched_tokens` for batch and queries dims.
- Reused across layers (backend overwrites these each layer call).

---

## 2. `PQCacheSpec` — `vllm/skylight/kv_cache_spec.py`

New `KVCacheSpec` subclass describing the per-block memory layout for the
PQ indexer cache.

### Fields

| Field | Type | Description |
|---|---|---|
| `block_size` | `int` | Number of tokens per block (inherited from `KVCacheSpec`) |
| `num_kv_heads` | `int` | Number of KV heads for the indexer cache (can be > 1) |
| `dtype` | `torch.dtype` | `torch.int8` — codebook indices stored as int8 |
| `pq_bits` | `int` | Bits per codebook index (codebook size = 2^pq_bits) |
| `group_factor` | `int` | Number of PQ sub-vectors per head |

### Methods / Properties

```python
@property
def page_size_bytes(self) -> int:
```
Returns the number of bytes per block. Formula:
`ceil(pq_bits / 8) * group_factor * num_kv_heads * block_size`.

---

## 3. `SkyLightAttentionBackend` — `vllm/skylight/attention_backend.py`

Lightweight vLLM framework backend. The vLLM framework calls into
`AttentionBackend` at three points:

| Framework call site | What it calls | What we provide |
|---|---|---|
| `init_attn_backend()` — group layers, create builder | `backend.get_builder_cls()` | `SkyLightMetadataBuilder` |
| `_reshape_kv_cache()` — allocate + shape cache tensor | `backend.get_kv_cache_shape()` | `(num_blocks, block_size, head_size)` |
| `build_attn_metadata()` — per-step metadata | `builder.build(common_attn_metadata)` | `SkyLightIndexerMetadata` |

### Fields

None — all methods are static/classmethod.

### Methods

```python
@staticmethod
def get_name() -> str:
```
Returns `"SKYLIGHT_INDEXER"`. Used for backend identification and grouping.

```python
@staticmethod
def get_builder_cls() -> type[SkyLightMetadataBuilder]:
```
Returns the `SkyLightMetadataBuilder` class. Called by the framework to
create per-group metadata builders.

```python
@staticmethod
def get_kv_cache_shape(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    cache_dtype_str: str = "auto",
) -> tuple[int, ...]:
```
Returns `(num_blocks, block_size, num_kv_heads, head_size)`. Called by the
framework to reshape the raw KV cache tensor. No constraint on `num_kv_heads`
— SkyLight indexer cache can have multiple KV heads.

```python
@staticmethod
def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
```
Returns `[MultipleOf(1)]` — no kernel block size constraint.

```python
@classmethod
def get_supported_head_sizes(cls) -> list[int]:
```
Returns `[]` — no head size constraint.

No `get_impl_cls()` needed — the indexer cache does not run attention forward.

---

## 4. `SkyLightIndexerMetadata` — `vllm/skylight/attention_backend.py`

Thin pass-through of `CommonAttentionMetadata` with a prefill/decode split.

### Fields

| Field | Type | Description |
|---|---|---|
| `slot_mapping` | `torch.Tensor` (int32) | `[num_actual_tokens]` — maps each token to its slot in the paged cache |
| `block_table` | `torch.Tensor` (int32) | `[num_reqs, max_blocks]` — maps each request to its physical cache blocks |
| `seq_lens` | `torch.Tensor` (int32) | `[num_reqs]` — total context length per request |
| `query_start_loc` | `torch.Tensor` (int32) | `[num_reqs + 1]` — cumulative query token offsets |
| `num_actual_tokens` | `int` | Total tokens in batch (excluding padding) |
| `num_decodes` | `int` | Number of decode requests in the batch |
| `num_decode_tokens` | `int` | Total tokens belonging to decode requests |
| `num_prefills` | `int` | Number of prefill requests in the batch |
| `num_prefill_tokens` | `int` | Total tokens belonging to prefill requests |

### Methods

None — pure data container produced by the metadata builder.

---

## 5. `SkyLightMetadataBuilder` — `vllm/skylight/attention_backend.py`

Builds `SkyLightIndexerMetadata` from the framework-provided
`CommonAttentionMetadata` each step.

### Fields

| Field | Type | Description |
|---|---|---|
| `kv_cache_spec` | `KVCacheSpec` | Spec for this cache group (from `__init__`) |
| `layer_names` | `list[str]` | Layer names sharing this builder (from `__init__`) |
| `vllm_config` | `VllmConfig` | Global config (from `__init__`) |
| `device` | `torch.device` | Target device (from `__init__`) |

Class variable:
- `reorder_batch_threshold: int = 1` — decodes have query_len <= 1

### Methods

```python
def __init__(
    self,
    kv_cache_spec: KVCacheSpec,
    layer_names: list[str],
    vllm_config: VllmConfig,
    device: torch.device,
) -> None:
```
Stores args via `super().__init__()`. No buffer pre-allocation needed.

```python
def build(
    self,
    common_prefix_len: int,
    common_attn_metadata: CommonAttentionMetadata,
    fast_build: bool = False,
) -> SkyLightIndexerMetadata:
```
1. Calls `split_decodes_and_prefills(common_attn_metadata)` to get
   `num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens`.
2. Extracts `slot_mapping`, `block_table_tensor`, `seq_lens`, `query_start_loc`
   from `common_attn_metadata`.
3. Returns a `SkyLightIndexerMetadata` with these fields.

---

## 6. `SkyLightIndexerCache` — `vllm/skylight/cache.py`

Base class for indexer caches. Implements `AttentionLayerBase` so vLLM
discovers it and allocates paged cache blocks for it.

### Fields

| Field | Type | Description |
|---|---|---|
| `prefix` | `str` | Unique layer name (e.g. `"model.layers.0.self_attn.indexer.k_cache"`) |
| `cache_config` | `CacheConfig` | Block size and cache config from vLLM |
| `kv_cache` | `torch.Tensor` | Paged cache tensor — initialized empty, populated by worker via `bind_kv_cache()` |

### Methods

```python
def __init__(self, prefix: str, cache_config: CacheConfig) -> None:
```
Stores `prefix`, `cache_config`. Initializes `kv_cache` as empty tensor.
Registers `self` in `compilation_config.static_forward_context[prefix]`.

```python
@abstractmethod
def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
```
Returns the cache spec describing per-block storage. Subclasses implement.

```python
def get_attn_backend(self) -> type[AttentionBackend]:
```
Returns `SkyLightAttentionBackend`. Shared by all SkyLight cache subclasses.

```python
def forward(self) -> None:
```
No-op. Required by `nn.Module` but the cache has no forward computation.

---

## 7. `SkyLightIndexerPQCache` — `vllm/skylight/cache.py`

PQ-specific cache subclass. Returns `PQCacheSpec` so vLLM allocates
appropriately-sized blocks for codebook storage.

### Fields

| Field | Type | Description |
|---|---|---|
| (inherited) | | All fields from `SkyLightIndexerCache` |
| `head_dim` | `int` | Key dimension per token |
| `dtype` | `torch.dtype` | Storage dtype (bf16/fp16) |
| `pq_bits` | `int` | Bits per codebook index |
| `group_factor` | `int` | Number of PQ sub-vectors |

### Methods

```python
def __init__(
    self,
    head_dim: int,
    dtype: torch.dtype,
    pq_bits: int,
    group_factor: int,
    prefix: str,
    cache_config: CacheConfig,
) -> None:
```
Calls `super().__init__(prefix, cache_config)`. Stores PQ-specific fields.

```python
def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
```
Returns `PQCacheSpec(block_size=..., num_kv_heads=1, head_size=self.head_dim,
dtype=self.dtype, pq_bits=self.pq_bits, group_factor=self.group_factor)`.

---

## 8. `SkyLightIndexerBackend` (ABC) — `vllm/skylight/backend.py`

Abstract base class for the compute backend. Defines the four operations
an indexer must support.

### Fields

| Field | Type | Description |
|---|---|---|
| `config` | `ResearchAttentionConfig` | Sparse attention config (contains masker configs) |

### Methods

```python
def __init__(self, config: ResearchAttentionConfig) -> None:
```
Stores `config`.

```python
@abstractmethod
def indexer_prefill(
    self,
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    metadata: SkyLightIndexerMetadata,
    sparse_index_buffers: SparseIndexBuffers,
    layer_idx: int,
) -> None:
```
Computes the sparsity pattern for prefill tokens and writes results into
`sparse_index_buffers`. During prefill the full key history is available
in `kv_cache`; the backend gathers keys using `metadata.block_table` and
`metadata.seq_lens`, scores them against `q`, and selects top-k.

- `q`: `[num_prefill_tokens, n_heads, head_dim]` — query vectors (after projection + RoPE)
- `kv_cache`: paged indexer key cache
- `metadata`: batch info (block_table, seq_lens, query_start_loc, etc.)
- `sparse_index_buffers`: output — write indices, weights, lens here
- `layer_idx`: which transformer layer (for per-layer PQ state)

```python
@abstractmethod
def indexer_decode(
    self,
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    metadata: SkyLightIndexerMetadata,
    sparse_index_buffers: SparseIndexBuffers,
    layer_idx: int,
) -> None:
```
Computes the sparsity pattern for decode tokens and writes results into
`sparse_index_buffers`. During decode, a single new key has already been
inserted into `kv_cache` by `kcache_decode`; the backend scores the query
against the full key history and selects top-k.

- `q`: `[num_decode_tokens, n_heads, head_dim]` — query vectors
- (rest same as `indexer_prefill`)

```python
@abstractmethod
def kcache_prefill(
    self,
    k: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    attn_kv_cache: torch.Tensor,
    attn_slot_mapping: torch.Tensor,
    metadata: SkyLightIndexerMetadata,
    layer_idx: int,
) -> None:
```
Builds the indexer cache for prefill tokens. Reads keys from the **attention
KV cache** (`attn_kv_cache`), quantizes them via PQ (kmeans clustering on
first call, incremental assignment on subsequent calls), and writes codebook
entries into the **indexer cache** (`kv_cache`). Also builds and stores
centroids on the backend object (`self.pq_centroids[layer_idx]`).

- `k`: `[num_prefill_tokens, head_dim]` — key vectors (for direct access without gathering)
- `kv_cache`: paged **indexer** cache tensor to write codebook entries into
- `slot_mapping`: `[num_prefill_tokens]` — target slot in indexer cache
- `attn_kv_cache`: paged **attention** KV cache (to gather full key history)
- `attn_slot_mapping`: `[num_prefill_tokens]` — slot mapping for the attention cache
- `metadata`: batch info (block_table, seq_lens for gathering from attn cache)
- `layer_idx`: which transformer layer (for per-layer centroid state)

```python
@abstractmethod
def kcache_decode(
    self,
    k: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    attn_kv_cache: torch.Tensor,
    attn_slot_mapping: torch.Tensor,
    metadata: SkyLightIndexerMetadata,
    layer_idx: int,
) -> None:
```
Builds the indexer cache for decode tokens. Reads the newly inserted key
from the **attention KV cache**, quantizes it using centroids built during
`kcache_prefill` (`self.pq_centroids[layer_idx]`), and writes the codebook
entry into the **indexer cache**.

- `k`: `[num_decode_tokens, head_dim]` — new key vectors
- `kv_cache`: paged **indexer** cache tensor to write codebook entry into
- `slot_mapping`: `[num_decode_tokens]` — target slot in indexer cache
- `attn_kv_cache`: paged **attention** KV cache
- `attn_slot_mapping`: `[num_decode_tokens]` — slot mapping for the attention cache
- `metadata`: batch info
- `layer_idx`: which transformer layer (accesses `self.pq_centroids[layer_idx]`)

---

## 9. `SkyLightPQBackendRef` — `vllm/skylight/pq_backend_ref.py`

Reference PyTorch implementation using `sparse_attention_hub` PQCache logic.

### Fields

| Field | Type | Description |
|---|---|---|
| `config` | `ResearchAttentionConfig` | Inherited from `SkyLightIndexerBackend` |
| `pq_config` | `PQCacheConfig` | Extracted from `config.masker_configs` — has `heavy_size`, `pq_group_factor`, `pq_bits`, `kmeans_iter`, `init_offset`, `metric` |
| `pq_centroids` | `dict[int, torch.Tensor \| None]` | Per-layer PQ centroids. Shape `[bsz, kv_heads, n_subvec, 2^pq_bits, subvec_d]`. `None` until first kmeans. |
| `pq_codebook` | `dict[int, torch.Tensor \| None]` | Per-layer codebooks. Shape `[bsz, n_quantized_keys, kv_heads, n_subvec]`. Grows with sequence. |
| `pq_ip2l2_phi` | `dict[int, torch.Tensor \| None]` | Per-layer IP→L2 augmentation scalars. Only used when `metric == "ip"`. |

### Methods

```python
def __init__(self, config: ResearchAttentionConfig) -> None:
```
Calls `super().__init__(config)`. Extracts `PQCacheConfig` from
`config.masker_configs`. Initializes `pq_centroids`, `pq_codebook`,
`pq_ip2l2_phi` as empty dicts.

```python
def kcache_prefill(
    self,
    k: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
```
Writes keys into paged cache in raw bf16/fp16. For each token, computes
`block_idx = slot // block_size`, `block_offset = slot % block_size`,
and stores `k` at `kv_cache[block_idx, block_offset, :]`.

```python
def kcache_decode(
    self,
    k: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
```
Same as `kcache_prefill` — stores decode keys into the paged cache.

```python
def indexer_prefill(
    self,
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    metadata: SkyLightIndexerMetadata,
    sparse_index_buffers: SparseIndexBuffers,
    layer_idx: int,
) -> None:
```
**Not implemented for now — raises `NotImplementedError`.**

During prefill, the full key set is available and attention can run dense.
Sparse index selection is only needed during decode. This can be implemented
later for long-context prefill optimization if needed.

```python
def indexer_decode(
    self,
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    metadata: SkyLightIndexerMetadata,
    sparse_index_buffers: SparseIndexBuffers,
    layer_idx: int,
) -> None:
```
Computes the sparsity pattern for decode tokens. Uses centroids from
`self.pq_centroids[layer_idx]` (built during `kcache_prefill`) and
codebook entries from `self.pq_codebook[layer_idx]` to score keys.

1. **PQ scoring**: computes `Q @ centroids` lookup table, gathers per-key
   approximate scores via codebook indices, sums across sub-vectors.
2. **Top-k selection**: `torch.topk()` on scores to select heavy hitters.
3. **Write output**: stores selected indices, their scores, and count into
   `sparse_index_buffers.indices`, `.weights`, `.lens`.

Requires that `kcache_prefill` has been called at least once for this
`layer_idx` so that `self.pq_centroids[layer_idx]` and
`self.pq_codebook[layer_idx]` are populated.

### PQ State Strategy

| State | Storage | Reason |
|---|---|---|
| `pq_centroids` | `dict` on backend object | Small (per-layer, fixed after kmeans) |
| `pq_codebook` | `dict` on backend object | Per-token, grows with sequence |
| `pq_ip2l2_phi` | `dict` on backend object | Small scalar per layer |
| `sparse_index_buffers` | `SparseIndexBuffers` on `SkyLightIndexer` | Shared across layers, reused each forward |

---

## 10. `SkyLightPQBackend` — `vllm/skylight/pq_backend.py`

Fast CUDA implementation (future work, placeholder for now).

### Fields

Same as `SkyLightPQBackendRef`.

### Methods

Same signatures as `SkyLightPQBackendRef` — CUDA kernel implementations
replacing the PyTorch reference logic.

```python
class SkyLightPQBackend(SkyLightIndexerBackend):
    # Same interface, CUDA kernel implementations
    ...
```

---

## 11. `SkyLightIndexer` — `vllm/skylight/indexer.py`

Top-level `nn.Module` that the model code instantiates. Orchestrates the
cache and compute backend.

### Fields

| Field | Type | Description |
|---|---|---|
| `vllm_config` | `VllmConfig` | Global vLLM config |
| `sparse_attention_config` | `ResearchAttentionConfig` | Sparse attention config from sparse_attention_hub |
| `backend` | `SkyLightIndexerBackend` | Compute backend instance (created from `backend_cls`) |
| `cache` | `SkyLightIndexerCache` | Indexer cache instance (manages paged KV cache) |
| `sparse_index_buffers` | `SparseIndexBuffers` | Shared output buffers |
| `prefix` | `str` | Layer name prefix |

### Methods

```python
def __init__(
    self,
    vllm_config: VllmConfig,
    sparse_attention_config: ResearchAttentionConfig,
    backend_cls: type[SkyLightIndexerBackend],
    cache: SkyLightIndexerCache,
    sparse_index_buffers: SparseIndexBuffers,
    prefix: str = "",
) -> None:
```
Stores all args. Instantiates `self.backend = backend_cls(sparse_attention_config)`.

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    layer_idx: int,
) -> SparseIndexBuffers:
```
Main entry point, called once per transformer layer.

- `hidden_states`: `[num_tokens, hidden_size]` — layer input (unused by indexer, passed for interface compatibility)
- `q`: `[num_tokens, n_heads, head_dim]` — query vectors (after projection + RoPE)
- `k`: `[num_tokens, head_dim]` — key vectors (after projection + RoPE)
- `layer_idx`: which transformer layer

Steps:
1. Retrieve `metadata: SkyLightIndexerMetadata` from
   `get_forward_context().attn_metadata[self.cache.prefix]`.
2. Get `kv_cache` from `self.cache.kv_cache`.
3. If `metadata.num_prefills > 0`:
   - Call `self.backend.kcache_prefill(k[decode_end:], kv_cache, slot_mapping[decode_end:])`.
   - Call `self.backend.indexer_prefill(q[decode_end:], kv_cache, metadata, self.sparse_index_buffers, layer_idx)`.
4. If `metadata.num_decodes > 0`:
   - Call `self.backend.kcache_decode(k[:decode_end], kv_cache, slot_mapping[:decode_end])`.
   - Call `self.backend.indexer_decode(q[:decode_end], kv_cache, metadata, self.sparse_index_buffers, layer_idx)`.
5. Return `self.sparse_index_buffers`.

---

## Design Summary

| Aspect | SkyLight PQ |
|---|---|
| Config | `ResearchAttentionConfig` from sparse_attention_hub |
| vLLM AttentionBackend | `SkyLightAttentionBackend` (lightweight) |
| Metadata builder | ~20 lines (split prefill/decode, pass through) |
| Metadata | 1 flat dataclass |
| Cache format | Raw bf16/fp16 keys + paged codebook |
| Cache spec | `PQCacheSpec` (stores pq_bits + group_factor) |
| Output buffers | `SparseIndexBuffers` triple (indices, weights, lens) with head dim |
| Scoring | PQ approximate scores (codebook lookup) |
| State | Stateful (centroids persist, codebook in cache) |
| Top-k | `torch.topk` (ref) / custom kernel (fast) |
| Projections | TBD — may reuse model's Q/K or have own |

---

## Mapping from PQCache (sparse_attention_hub) to Backend Methods

| PQCache method | Maps to |
|---|---|
| `_perform_kmeans_clustering()` | `indexer_prefill` (first call) |
| `_handle_incremental_keys()` | `indexer_decode` (codebook update) |
| `_compute_pq_scores()` | `indexer_prefill` / `indexer_decode` (scoring) |
| `_create_pq_mask()` (top-k) | `indexer_prefill` / `indexer_decode` (selection) |
| N/A (new) | `kcache_prefill` / `kcache_decode` (paged cache write) |

---

## Next Steps

1. Define `PQCacheSpec` in `vllm/skylight/kv_cache_spec.py`
2. Define `SparseIndexBuffers` in `vllm/skylight/buffers.py`
3. Implement `SkyLightAttentionBackend` + `SkyLightMetadataBuilder` + `SkyLightIndexerMetadata` in `vllm/skylight/attention_backend.py`
4. Implement `SkyLightIndexerCache` + `SkyLightIndexerPQCache` in `vllm/skylight/cache.py`
5. Implement `SkyLightIndexerBackend` ABC in `vllm/skylight/backend.py`
6. Implement `SkyLightPQBackendRef` in `vllm/skylight/pq_backend_ref.py`
7. Implement `SkyLightIndexer` in `vllm/skylight/indexer.py`
8. Wire into model code
