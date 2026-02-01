# SKLT Sparse Attention Backend Implementation Plan

## Overview
Implementation of SKLT (SkyLight) sparse attention backend for vLLM, supporting arbitrary per-head sparse masks with configurable indexers.

## Architecture

### Core Components

1. **sklt_sparse_attention**: PyTorch kernel for weighted sparse attention computation
2. **Indexer System**: Abstract base class with pluggable implementations
   - Base: `BaseIndexer` - abstract interface
   - Impl: `StreamingIndexer` - sink tokens + local window pattern
3. **Backend Integration**: Full vLLM attention backend (Backend, Impl, MetadataBuilder)

## Directory Structure

```
vllm/v1/attention/backends/sklt/
├── __init__.py
├── sklt_backend.py              # SKLTAttentionBackend class
├── sklt_impl.py                 # SKLTAttentionImpl class
├── sklt_metadata.py             # SKLTAttentionMetadata & Builder
├── indexer/
│   ├── __init__.py
│   ├── base_indexer.py          # BaseIndexer abstract class
│   └── streaming_indexer.py    # StreamingIndexer implementation
└── ops/
    ├── __init__.py
    └── sparse_attention.py      # sklt_sparse_attention PyTorch kernel

tests/v1/attention/
├── test_sklt_backend.py         # Backend integration tests
└── test_sklt_indexer.py         # Indexer unit tests

vllm/config/
└── attention.py                 # Add IndexerConfig
```

## Data Structures

### SparsityInfo
Container for per-head sparse attention patterns:
- `sparse_len`: `(B, queries, num_query_heads, 1)` - number of valid keys per query head
- `sparse_idx`: `(B, queries, num_query_heads, max_sparse_k)` - indices of sparse keys
- `sparse_weights`: `(B, queries, num_query_heads, max_sparse_k)` - attention weights per sparse key

**Key Properties:**
- Pre-allocated buffers for CUDA graph compatibility
- Only first `sparse_len[b,q,h]` entries are valid in `sparse_idx` and `sparse_weights`
- Supports different sparsity patterns per query head
- Weights can be arbitrary (not just 0/1 mask)

### IndexerConfig
Configuration for sparse indexer behavior:
```python
indexer_type: str               # "streaming", "custom", etc.
num_sink_tokens: int           # Streaming: number of sink tokens
local_window_size: int         # Streaming: local attention window
max_sparse_k: int              # Maximum sparse keys per head (buffer size)
```

Integrated into `vllm.config.AttentionConfig`:
- `indexer_config: Optional[IndexerConfig]`
- `use_sklt_attention: bool`

## Implementation Phases

### Phase 1: Configuration Setup

**Goal:** Add indexer configuration to vLLM config system

**Files to Modify:**
- `vllm/config/attention.py`: Add `IndexerConfig` dataclass and integrate into `AttentionConfig`

**Deliverables:**
- `IndexerConfig` with streaming indexer parameters
- Config validation logic
- Integration with existing attention config

### Phase 2: Indexer System

**Goal:** Implement abstract indexer interface and streaming indexer

#### 2.1 Base Indexer (`base_indexer.py`)

**Class:** `SparsityInfo`
- Dataclass for sparse pattern storage
- Shape validation method
- Device management

**Class:** `BaseIndexer(ABC)`
- Abstract base class for all indexers
- Constructor: `__init__(indexer_config, device)`
- Abstract method: `_init_buffers()` - pre-allocate CUDA graph compatible buffers
- Abstract method: `compute_sparsity(...)` - generate sparsity information
- Abstract method: `get_max_sparse_k()` - return max sparse keys per head
- Buffer management for reuse across iterations

**Key Design Points:**
- Buffers allocated once at initialization (max batch size from scheduler config)
- Return sliced views for actual batch size in `compute_sparsity`
- Thread-safe if needed for multi-layer support (future)

#### 2.2 Streaming Indexer (`streaming_indexer.py`)

**Class:** `StreamingIndexer(BaseIndexer)`

**Sparsity Pattern:**
For each query position `q` at sequence position `ctx_len + q`:
1. **Sink tokens**: First `K` tokens (positions `0` to `K-1`)
2. **Local window**: Last `W` tokens before query (positions `max(K, pos-W)` to `pos-1`)
3. **Current token**: Position `pos` (causal)

**Implementation Details:**
- Pure PyTorch implementation (no custom CUDA/Triton for now)
- Compute pattern for each (batch, query, head) combination
- Set all `sparse_weights` to 1.0 (uniform weighting)
- Same sparsity pattern across all query heads (can be different per head in future)
- Handle edge cases: sequence shorter than window, overlapping sink/window

**Compute Flow:**
```python
for each sequence in batch:
    for each query position:
        indices = []
        # Add sink tokens
        indices += [0, 1, ..., min(K-1, pos)]
        # Add local window
        indices += [max(K, pos-W), ..., pos-1, pos]
        # Remove duplicates, sort
        # Populate sparse_idx, sparse_len
        # Set sparse_weights = 1.0
```

**Deliverables:**
- Buffer initialization with proper sizing
- `compute_sparsity()` implementation
- Validation and edge case handling

### Phase 3: Sparse Attention Kernel

**Goal:** Implement PyTorch sparse attention computation

**File:** `ops/sparse_attention.py`

**Function:** `sklt_sparse_attention(...)`

**Signature:**
```python
def sklt_sparse_attention(
    query: torch.Tensor,           # (num_tokens, num_heads, head_size)
    key_cache: torch.Tensor,       # KV cache (backend-specific shape)
    value_cache: torch.Tensor,     # KV cache (backend-specific shape)
    sparsity_info: SparsityInfo,   # Sparse pattern information
    block_table: torch.Tensor,     # (batch_size, max_blocks)
    query_start_loc: torch.Tensor, # (batch_size + 1,)
    seq_lens: torch.Tensor,        # (batch_size,)
    block_size: int,
    scale: float,
    num_kv_heads: int,             # For GQA support
    output: torch.Tensor,          # Pre-allocated output
) -> torch.Tensor:
```

**Algorithm:**
```python
for each batch:
    for each query token:
        for each query head:
            # 1. Extract sparse pattern
            sparse_k = sparse_len[b, q, h]
            indices = sparse_idx[b, q, h, :sparse_k]
            weights = sparse_weights[b, q, h, :sparse_k]
            
            # 2. Gather K, V from cache using indices
            for idx in indices:
                block_idx = idx // block_size
                block_offset = idx % block_size
                physical_block = block_table[b, block_idx]
                kv_head = h // (num_query_heads // num_kv_heads)  # GQA
                k[...] = key_cache[physical_block, block_offset, kv_head]
                v[...] = value_cache[physical_block, block_offset, kv_head]
            
            # 3. Compute attention scores
            scores = (Q[h] @ K.T) * scale  # (sparse_k,)
            
            # 4. Apply sparse weights (element-wise multiply)
            scores = scores * weights
            
            # 5. Softmax over sparse keys
            attn_probs = softmax(scores)  # (sparse_k,)
            
            # 6. Weighted sum of values
            output[token, h] = attn_probs @ V  # (head_size,)
```

**Key Features:**
- Support arbitrary per-head weights (not just binary masks)
- Proper GQA/MQA support (map query heads to KV heads)
- KV cache access via block table (paged attention)
- Causal masking handled by indexer (not in kernel)

**Implementation Notes:**
- Pure PyTorch (nested loops acceptable for correctness)
- Assume FlashAttention-style KV cache layout: `(2, num_blocks, block_size, num_kv_heads, head_size)`
- Output buffer pre-allocated (avoid allocation in forward pass)
- No optimization focus - clarity and correctness first

### Phase 4: Backend Integration

**Goal:** Integrate into vLLM attention backend system

#### 4.1 Metadata (`sklt_metadata.py`)

**Class:** `SKLTAttentionMetadata`
- Extends vLLM metadata pattern
- Contains `SparsityInfo` object
- Standard fields: num_tokens, seq_lens, block_table, slot_mapping, etc.
- Optional: cascade attention support (future)

**Class:** `SKLTAttentionMetadataBuilder(AttentionMetadataBuilder)`
- Initialize indexer in `__init__` from config
- `_cudagraph_support = AttentionCGSupport.ALWAYS`
- `build()` method:
  1. Extract common metadata
  2. Call `indexer.compute_sparsity(...)`
  3. Package into `SKLTAttentionMetadata`

**Indexer Lifecycle:**
- Single global indexer instance per builder
- Created in `__init__` based on `vllm_config.attention_config.indexer_config`
- Reused across all `build()` calls (buffer reuse)
- Factory pattern for different indexer types

#### 4.2 Implementation (`sklt_impl.py`)

**Class:** `SKLTAttentionImpl(AttentionImpl)`
- Standard `__init__`: store num_heads, head_size, scale, etc.
- `forward()` method:
  1. Extract metadata
  2. Unbind KV cache into key_cache, value_cache
  3. Call `sklt_sparse_attention(...)`
  4. Return output
- Handle prefill vs decode uniformly (indexer decides pattern)
- KV cache update: use standard reshape_and_cache (no changes needed)

**Features:**
- GQA/MQA support via num_kv_heads
- Support for sliding window (future - indexer decides)
- No special cascade attention handling (Phase 1)

#### 4.3 Backend (`sklt_backend.py`)

**Class:** `SKLTAttentionBackend(AttentionBackend)`

**Static Methods:**
- `get_name()` → `"SKLT"`
- `get_impl_cls()` → `SKLTAttentionImpl`
- `get_builder_cls()` → `SKLTAttentionMetadataBuilder`
- `get_kv_cache_shape()` → Standard FlashAttention layout
- `get_supported_kernel_block_sizes()` → `[16, 32, 64, 128]`

**Validation Methods:**
- `supports_head_size()`: `[64, 80, 96, 128, 256]`
- `supports_dtype()`: `[fp16, bf16]`
- `supports_kv_cache_dtype()`: `["auto", "bfloat16"]`
- `supports_compute_capability()`: `>= 7.0` (for initial PyTorch version)
- `is_sparse()` → `True`

**Configuration Validation:**
- Require `indexer_config` to be set
- Validate max_sparse_k vs actual pattern sizes

### Phase 5: Registration & Testing

#### 5.1 Backend Registration

**File:** `vllm/v1/attention/backends/registry.py`
- Add `SKLT` enum entry with path to `SKLTAttentionBackend`

**File:** `vllm/v1/attention/backends/__init__.py`
- Export SKLT backend classes

#### 5.2 Unit Tests

**File:** `tests/v1/attention/test_sklt_indexer.py`

**Tests:**
- `test_sparsity_info_shapes`: Validate SparsityInfo structure
- `test_streaming_indexer_sink_only`: Only sink tokens (window=0)
- `test_streaming_indexer_window_only`: Only local window (sink=0)
- `test_streaming_indexer_combined`: Sink + window, verify no duplicates
- `test_streaming_indexer_short_sequence`: Sequence shorter than window
- `test_streaming_indexer_per_head`: Verify consistent pattern across heads
- `test_indexer_buffer_reuse`: Multiple calls reuse buffers correctly
- `test_max_sparse_k`: Verify max_sparse_k calculation

**File:** `tests/v1/attention/test_sklt_backend.py`

**Tests:**
- `test_backend_registration`: SKLT in enum, correct paths
- `test_metadata_builder`: Build metadata from common metadata
- `test_sparse_attention_single_head`: Correctness vs reference attention
- `test_sparse_attention_gqa`: GQA with different num_query_heads/num_kv_heads
- `test_sparse_attention_batched`: Multiple sequences in batch
- `test_weighted_attention`: Verify weights are applied correctly
- `test_backend_validation`: Config validation works

#### 5.3 Integration Test

**Goal:** End-to-end test with simple model

**Test:** `test_sklt_e2e`
- Use small test model (e.g., tiny GPT-2)
- Configure with SKLT backend
- Run inference with streaming indexer
- Compare output with FlashAttention backend (should be different but valid)
- Verify no crashes, proper shapes

## Design Decisions & Rationale

### 1. Arbitrary Weights Support
**Decision:** `sklt_sparse_attention` supports arbitrary weights, but `StreamingIndexer` sets weights=1.0

**Rationale:**
- Future indexers may need non-uniform weights (e.g., learned weights, distance-based decay)
- Kernel complexity is the same (element-wise multiply)
- Indexer simplicity for initial implementation

### 2. Per-Head Sparsity Patterns
**Decision:** Support different patterns per query head from the start

**Rationale:**
- Some architectures may benefit from head-specific patterns (future)
- Memory layout supports it naturally
- StreamingIndexer uses same pattern across heads (simple), but infrastructure ready

### 3. PyTorch-First Approach
**Decision:** Pure PyTorch implementation, no Triton/CUDA kernels in Phase 1

**Rationale:**
- Correctness validation before optimization
- Easier debugging and iteration
- Performance optimization is Phase 2 (separate effort)

### 4. Global Indexer
**Decision:** One indexer instance per MetadataBuilder (effectively global)

**Rationale:**
- Simpler initial design
- Buffer reuse across iterations
- Per-layer indexers can be added later via config extension

### 5. CUDA Graph Compatibility
**Decision:** Pre-allocate buffers at max size, slice for actual use

**Rationale:**
- CUDA graphs require static memory addresses
- Performance critical for vLLM
- Standard pattern in existing backends (FlashAttention, Triton)

### 6. KV Cache Layout
**Decision:** Use standard FlashAttention layout

**Rationale:**
- Compatible with existing vLLM infrastructure
- No need for custom cache management
- Leverage existing reshape_and_cache operations

## Success Criteria

### Phase 1-3 (Indexer + Kernel)
- [ ] `StreamingIndexer` produces correct sparse patterns
- [ ] Sparse patterns match expected sink + window logic
- [ ] `sklt_sparse_attention` computes correct attention with arbitrary weights
- [ ] Unit tests pass for various configurations

### Phase 4-5 (Backend Integration)
- [ ] Backend registers successfully in vLLM
- [ ] Metadata builder creates valid metadata
- [ ] End-to-end inference runs without errors
- [ ] Output shapes and dtypes correct
- [ ] Configuration validation catches invalid configs

### Correctness Validation
- [ ] For uniform weights, output matches reference attention on sparse subset
- [ ] For non-uniform weights, weighted attention is computed correctly
- [ ] GQA/MQA mapping works correctly
- [ ] Batch processing handles variable sequence lengths

## Future Extensions (Post Phase 5)

1. **Performance:** Triton/CUDA kernels for sparse attention
2. **Indexers:** 
   - Block-sparse patterns
   - Learned/adaptive sparsity
   - Per-layer indexers
3. **Features:**
   - FP8 KV cache support
   - Cascade attention for common prefixes
   - Integration with chunked prefill
4. **Optimization:**
   - Fused indexer + attention kernel
   - Optimized gather operations
   - Better memory layout for sparse access

## Open Questions / TODOs

1. **Buffer Sizing:** What's the right max_batch_size for buffer allocation? (Use scheduler config max_num_seqs?)
2. **GQA Edge Cases:** Verify KV head indexing for all GQA ratios (1:1, 2:1, 4:1, 8:1, etc.)
3. **Validation:** Should indexer validate that sparse_k <= max_sparse_k and error/warn?
4. **Debugging:** Add logging/debug mode to dump sparsity patterns for inspection?
5. **Config Defaults:** What are sensible defaults for num_sink_tokens and local_window_size?

## Implementation Order

1. **Week 1:** Phase 1 (Config) + Phase 2.1 (Base Indexer)
2. **Week 1-2:** Phase 2.2 (StreamingIndexer) + Unit Tests
3. **Week 2:** Phase 3 (Sparse Attention Kernel) + Unit Tests
4. **Week 3:** Phase 4 (Backend Integration)
5. **Week 3-4:** Phase 5 (Registration, Testing, E2E Validation)

## References

- vLLM FlashAttention Backend: `vllm/v1/attention/backends/flash_attn.py`
- vLLM Triton Backend: `vllm/v1/attention/backends/triton_attn.py`
- Attention Backend Base: `vllm/v1/attention/backend.py`
- Existing Indexer Example: `vllm/v1/attention/backends/mla/indexer.py` (DeepSeek)
