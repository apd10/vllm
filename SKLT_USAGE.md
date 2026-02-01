# SKLT Sparse Attention Backend - Usage Guide

## Overview

SKLT (SkyLight) is a sparse attention backend for vLLM that supports arbitrary per-head sparse masks. The current implementation includes a **StreamingIndexer** that efficiently handles sink tokens and local attention windows.

## Features

- **Sparse Attention**: Only attend to a subset of keys, reducing computation
- **Streaming Pattern**: Combines attention sinks (first K tokens) with local window (last W tokens)
- **Per-Head Patterns**: Supports different sparsity patterns for each attention head
- **Weighted Attention**: Supports arbitrary weights (not just binary masks)
- **CUDA Graph Compatible**: Pre-allocated buffers for efficient execution
- **Decode Optimized**: Current implementation is optimized for decode phase (single query tokens)

## Configuration

### IndexerConfig

```python
from vllm.config.attention import IndexerConfig

indexer_config = IndexerConfig(
    indexer_type="streaming",     # Type of indexer (currently: "streaming")
    num_sink_tokens=4,            # Number of initial tokens to always attend to
    local_window_size=512,        # Size of local attention window
    max_sparse_k=1024,            # Maximum sparse keys per head (buffer size)
)
```

### AttentionConfig

```python
from vllm.config.attention import AttentionConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum

attention_config = AttentionConfig(
    backend=AttentionBackendEnum.SKLT,
    indexer_config=indexer_config,
    use_sparse_attention=True,
)
```

## Usage Examples

### 1. Python API

```python
from vllm import LLM, SamplingParams
from vllm.config.attention import AttentionConfig, IndexerConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum

# Configure SKLT backend
indexer_config = IndexerConfig(
    indexer_type="streaming",
    num_sink_tokens=4,
    local_window_size=512,
    max_sparse_k=1024,
)

attention_config = AttentionConfig(
    backend=AttentionBackendEnum.SKLT,
    indexer_config=indexer_config,
    use_sparse_attention=True,
)

# Create LLM with SKLT backend
llm = LLM(
    model="facebook/opt-125m",
    attention_config=attention_config,
    max_model_len=1024,
    enforce_eager=True,
)

# Generate text
prompts = ["Hello, my name is"]
sampling_params = SamplingParams(temperature=0.8, max_tokens=50)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)
```

### 2. Command Line (vllm serve)

```bash
# Activate vLLM conda environment
/workspace/anaconda3/envs/vllm/bin/python

# Run with SKLT backend
python -m vllm.entrypoints.openai.api_server \
    --model facebook/opt-125m \
    --attention-backend SKLT \
    --attention-config '{"indexer_config": {"indexer_type": "streaming", "num_sink_tokens": 4, "local_window_size": 512, "max_sparse_k": 1024}, "use_sparse_attention": true}' \
    --enforce-eager
```

Or use the structured config format:

```bash
/workspace/anaconda3/envs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
    --model facebook/opt-125m \
    -ac.backend SKLT \
    -ac.use_sparse_attention true \
    -ac.indexer_config.indexer_type streaming \
    -ac.indexer_config.num_sink_tokens 4 \
    -ac.indexer_config.local_window_size 512 \
    -ac.indexer_config.max_sparse_k 1024 \
    --enforce-eager
```

### 3. Example Script

Run the provided example:

```bash
/workspace/anaconda3/envs/vllm/bin/python examples/sklt_example.py
```

## Streaming Indexer Pattern

For each query at position `q`, the StreamingIndexer creates a sparse pattern:

1. **Sink Tokens**: First `num_sink_tokens` tokens (positions 0 to K-1)
2. **Local Window**: Last `local_window_size` tokens before query (positions max(K, q-W) to q)
3. **Union**: All unique positions from sink ∪ window

Example with `num_sink_tokens=4`, `local_window_size=8`:
- Query at position 20
- Sink: [0, 1, 2, 3]
- Window: [12, 13, 14, 15, 16, 17, 18, 19, 20]
- Sparse pattern: [0, 1, 2, 3, 12, 13, 14, 15, 16, 17, 18, 19, 20]

## Performance Considerations

### Current Implementation (PyTorch)

The current implementation uses pure PyTorch for correctness validation:
- ✅ Correct sparse attention computation
- ✅ Arbitrary per-head weights
- ✅ Works on any CUDA device (SM 7.0+)
- ⚠️ Not optimized for performance (nested loops)
- ⚠️ **Decode-only**: Optimized for single query tokens (decode phase)
- ⚠️ **Prefill**: Works but may be slow with multiple queries; use different backend for prefill

### Future Optimizations (Planned)

- Triton kernels for fused indexing + attention
- CUDA kernels for optimized sparse gather/scatter
- Fused KV cache lookup
- Better memory layout for sparse access

## Troubleshooting

### Out of Memory Errors

If you see OOM errors during initialization, reduce buffer sizes in `streaming_indexer.py`:

```python
# In streaming_indexer.py, _init_buffers()
max_batch = 256      # Reduce from 512
max_queries = 2048   # Reduce from 8192
max_heads = 64       # Reduce from 128
```

Or reduce `max_sparse_k` in your config:

```python
indexer_config = IndexerConfig(
    max_sparse_k=512,  # Reduce from 1024
    ...
)
```

### Backend Validation Errors

Ensure you have:
- `use_sparse_attention=True` in `AttentionConfig`
- Valid `indexer_config` in `AttentionConfig`
- CUDA device with compute capability >= 7.0

### Performance Warnings

If you see warnings about `max_query_len > 1`:
- This means SKLT is being used during prefill (multiple query tokens)
- Current PyTorch implementation is not optimized for this
- Consider using chunked prefill or a different backend for prefill phase
- Decode phase (single query tokens) will still benefit from sparse attention

## Testing

Run unit tests:

```bash
/workspace/anaconda3/envs/vllm/bin/python -m pytest tests/v1/attention/test_sklt_backend.py -v
```

Run simple validation:

```bash
/workspace/anaconda3/envs/vllm/bin/python test_sklt_simple.py
```

## Implementation Details

### Directory Structure

```
vllm/v1/attention/backends/sklt/
├── __init__.py
├── sklt_backend.py              # Backend class
├── sklt_impl.py                 # Attention implementation
├── sklt_metadata.py             # Metadata builder
├── indexer/
│   ├── __init__.py
│   ├── base_indexer.py          # Abstract indexer interface
│   └── streaming_indexer.py    # Streaming indexer implementation
└── ops/
    ├── __init__.py
    └── sparse_attention.py      # PyTorch sparse attention kernel
```

### Key Classes

- **SKLTAttentionBackend**: Backend registration and validation
- **SKLTAttentionImpl**: Forward pass implementation
- **SKLTAttentionMetadataBuilder**: Builds metadata with sparsity info
- **BaseIndexer**: Abstract indexer interface
- **StreamingIndexer**: Sink + window pattern implementation
- **SparsityInfo**: Container for sparse patterns (indices, lengths, weights)

## Future Extensions

1. **Additional Indexers**:
   - Block-sparse patterns
   - Learned/adaptive sparsity
   - Per-layer indexers
   
2. **Performance Optimizations**:
   - Triton/CUDA kernels
   - Fused operations
   - Better memory layouts
   
3. **Features**:
   - FP8 KV cache support
   - Cascade attention
   - Encoder-decoder support

## References

- vLLM Documentation: https://docs.vllm.ai
- Implementation Plan: `/workspace/vllm/PLAN.md`
- Example Script: `/workspace/vllm/examples/sklt_example.py`
