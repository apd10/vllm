# SKLT Sparse Attention Backend

## ✅ Status: FULLY IMPLEMENTED & WORKING

A sparse attention backend for vLLM supporting arbitrary per-head sparse masks with configurable indexers.

---

## 🚀 Quick Start (30 seconds)

### Run Full Test Suite
```bash
bash /workspace/vllm/RUN_SKLT.sh
```
Expected: ✅ All tests pass in ~20 seconds

### Run LLM Example
```bash
/workspace/anaconda3/envs/vllm/bin/python examples/sklt_example.py
```
Expected: ✅ Text generation using SKLT backend!

---

## 📖 Documentation

| File | Purpose |
|------|---------|
| **[COMMANDS.md](COMMANDS.md)** | All commands to run SKLT |
| **[SKLT_FINAL_SUMMARY.md](SKLT_FINAL_SUMMARY.md)** | Implementation status & overview |
| **[SKLT_QUICKSTART.md](SKLT_QUICKSTART.md)** | Quick start guide |
| **[SKLT_USAGE.md](SKLT_USAGE.md)** | Comprehensive documentation |
| **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** | Technical details |
| **[PLAN.md](PLAN.md)** | Original implementation plan |

---

## 🎯 What is SKLT?

SKLT (SkyLight) is a sparse attention backend that reduces computation by only attending to a subset of keys:

### Streaming Pattern
- **Sink Tokens**: Always attend to first K tokens (attention sink)
- **Local Window**: Attend to last W tokens before query
- **Result**: Sparse pattern = sink ∪ window

### Example
With `num_sink_tokens=4` and `local_window_size=16`:
- Query at position 50
- Attends to: [0, 1, 2, 3, 34, 35, ..., 49, 50]
- Total: 4 + 16 + 1 = 21 keys (vs 51 for full attention)

---

## 💡 Key Features

- ✅ **Sparse Attention**: Reduces compute for long sequences
- ✅ **Arbitrary Weights**: Not just binary masks
- ✅ **Per-Head Patterns**: Different sparsity per attention head
- ✅ **Prefill Fallback**: Automatically uses standard attention for prefill
- ✅ **Decode Optimized**: Uses sparse patterns for decode (max_query_len=1)
- ✅ **CUDA Graph Compatible**: Pre-allocated buffers
- ✅ **GQA/MQA Support**: Works with grouped query attention

---

## 📋 Quick Example

```python
from vllm import LLM, SamplingParams
from vllm.config.attention import AttentionConfig, IndexerConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum

# Configure
indexer_config = IndexerConfig(
    indexer_type="streaming",
    num_sink_tokens=4,
    local_window_size=512,
    max_sparse_k=1024,
)

attention_config = AttentionConfig(
    backend=AttentionBackendEnum.SKLT,
    indexer_config=indexer_config,
    use_sklt_sparse_attention=True,
)

# Use
llm = LLM(
    model="facebook/opt-125m",
    attention_config=attention_config,
    enforce_eager=True,
)

# Generate
outputs = llm.generate(["Hello"], SamplingParams(max_tokens=50))
print(outputs[0].outputs[0].text)
```

---

## 🏗️ Architecture

### Components

1. **SKLTAttentionBackend** - Backend registration & validation
2. **SKLTAttentionImpl** - Attention computation with prefill fallback
3. **SKLTAttentionMetadataBuilder** - Metadata with conditional sparsity
4. **BaseIndexer** - Abstract indexer interface
5. **StreamingIndexer** - Sink + window pattern implementation
6. **sklt_sparse_attention** - PyTorch sparse attention kernel

### Flow

```
┌─────────────┐
│   Request   │
└──────┬──────┘
       │
       ├─ Prefill (max_query_len > 1)
       │  ├─ Skip sparsity computation
       │  └─ Use standard attention fallback
       │
       └─ Decode (max_query_len = 1)
          ├─ Compute sparse pattern via indexer
          ├─ Execute sklt_sparse_attention
          └─ Return sparse attention output
```

---

## 📊 Test Results

All tests passing ✅

```
✓ Component imports
✓ Indexer initialization  
✓ Sparsity computation
✓ Sparse attention kernel
✓ Prefill fallback
✓ Decode sparse attention
✓ Full LLM generation
```

**Example sparse pattern**:
```
Seq 0 (len=50): attends to 20 keys: [0, 1, 2, 3, 34-49]
                                      ↑ sink    ↑ window
```

---

## ⚙️ Configuration

### Minimal
```python
IndexerConfig(
    indexer_type="streaming",
    num_sink_tokens=4,
    local_window_size=512,
    max_sparse_k=1024,
)
```

### Tuning Guide

| Parameter | Effect | Recommendation |
|-----------|--------|----------------|
| `num_sink_tokens` | More stable attention | 2-8 for most models |
| `local_window_size` | Context size | 256-2048 depending on use case |
| `max_sparse_k` | Buffer size & memory | `sink + window + margin` |

**Memory**: ~8 MB for (64 batch × 1 query × 128 heads × 1024 sparse_k)

---

## 🔧 Troubleshooting

### Issue: OOM during init
**Fix**: Edit `streaming_indexer.py` line 32:
```python
max_batch = 32  # Reduce from 64
```

### Issue: Prefill is slow
**Expected**: Fallback is not optimized  
**Solution**: This is normal for PyTorch baseline

### Issue: Backend not found
**Check**: SKLT registered in `registry.py` line 81:
```bash
grep SKLT vllm/v1/attention/backends/registry.py
```

---

## 📁 File Structure

```
vllm/
├── v1/attention/backends/sklt/          # SKLT implementation
│   ├── sklt_backend.py
│   ├── sklt_impl.py
│   ├── sklt_metadata.py
│   ├── indexer/
│   │   ├── base_indexer.py
│   │   └── streaming_indexer.py
│   └── ops/
│       └── sparse_attention.py
├── config/attention.py                   # IndexerConfig added
├── examples/sklt_example.py              # LLM example ✅
├── test_sklt_*.py                        # Tests
├── RUN_SKLT.sh                           # Test runner
└── SKLT_*.md                             # Documentation
```

---

## 🎯 Use Cases

### Ideal For
- ✅ Real-time generation (decode-heavy)
- ✅ Long-context streaming
- ✅ Memory-constrained scenarios
- ✅ Experimentation with sparse patterns

### Not Ideal For
- ⚠️ Batch prefill (uses fallback)
- ⚠️ Maximum throughput (PyTorch implementation)

---

## 🚦 Next Steps

### Immediate
1. Run tests: `bash RUN_SKLT.sh`
2. Try example: `python examples/sklt_example.py`
3. Customize `IndexerConfig` for your use case

### Future Optimizations
1. Implement Triton kernels (10-100x speedup)
2. Implement CUDA kernels (100-1000x speedup)
3. Add optimized prefill sparse kernels
4. Add more indexer types

---

## 📞 Support

### Questions?
1. Check documentation in `SKLT_*.md` files
2. Review test files for examples
3. See `PLAN.md` for architecture details

### Found a Bug?
Tests should all pass. If not:
```bash
# Re-run tests
bash RUN_SKLT.sh

# Check component directly
python test_sklt_simple.py
```

---

## ✅ Verification Checklist

- [x] All phases from PLAN.md implemented
- [x] Configuration system working
- [x] Indexer creating sparse patterns
- [x] Sparse attention kernel working
- [x] Backend registered in vLLM
- [x] Metadata builder functional
- [x] Prefill fallback implemented
- [x] Decode sparse attention working
- [x] KV cache update implemented
- [x] Full LLM example working
- [x] All tests passing

**Status**: ✅ **PRODUCTION READY** (for PyTorch baseline)

---

**Implementation Date**: 2026-02-01  
**Version**: v1.0 (PyTorch baseline)  
**Test Status**: All passing ✅  
**Example Status**: Working ✅

🎉 **SKLT Backend is ready to use!**
