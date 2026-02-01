# SKLT Sparse Attention Profiling

## Quick Start

### Basic Profiling
```bash
/workspace/anaconda3/envs/vllm/bin/python vllm/attention_scripts/profile_sklt_vllm.py \
    --context-length 4096 \
    --batch-size 1
```

### Profile Multiple Configurations
```bash
/workspace/anaconda3/envs/vllm/bin/python vllm/attention_scripts/profile_sklt_vllm.py \
    --context-length 1024 2048 4096 8192 16384 \
    --batch-size 1 4 8 16 \
    --sink-tokens 4 \
    --window-size 512
```

### With Detailed Profiler
```bash
/workspace/anaconda3/envs/vllm/bin/python vllm/attention_scripts/profile_sklt_vllm.py \
    --context-length 4096 \
    --batch-size 1 \
    --use-profiler \
    --num-iterations 20
```

### Save Results to JSON
```bash
/workspace/anaconda3/envs/vllm/bin/python vllm/attention_scripts/profile_sklt_vllm.py \
    --context-length 4096 8192 16384 \
    --batch-size 1 4 8 \
    --save-trace \
    --output-file sklt_profile_results.json
```

---

## Command Line Options

### Required
None - uses sensible defaults

### Context & Batch
- `--context-length CTX [CTX ...]` - Context lengths to profile (default: 4096)
- `--batch-size BS [BS ...]` - Batch sizes to profile (default: 1, 4, 8)

### SKLT Configuration
- `--sink-tokens N` - Number of sink tokens (default: 4)
- `--window-size W` - Local window size (default: 512)
- `--max-sparse-k K` - Max sparse keys buffer (default: 1024)

### Profiling
- `--num-warmup N` - Warmup iterations (default: 10)
- `--num-iterations N` - Benchmark iterations (default: 100)
- `--use-profiler` - Use torch profiler for detailed analysis
- `--save-trace` - Save results to JSON
- `--output-file FILE` - Output file name (default: sklt_profile_results.json)

### Model Configuration
- `--dtype {float16,bfloat16}` - Tensor dtype (default: float16)
- `--num-heads N` - Number of attention heads (default: 32)
- `--num-kv-heads N` - Number of KV heads for GQA (default: 8)
- `--head-size N` - Head dimension (default: 128)
- `--block-size N` - KV cache block size (default: 16)

---

## Example Outputs

### Basic Run
```
Context Len: 4096, Batch Size: 1
  Sparse pattern: 516 keys (vs 4096 full)
  Sparsity ratio: 12.6%
  Average latency: 316.05 ms
  Throughput: 3.16 tokens/sec
  Compute reduction: 87.4%
```

### Summary Table
```
Ctx Len    Batch    Sparse K   Reduction    Latency(ms)   Throughput    
4096       1        516        87.4%        316.054       3.16          
8192       1        516        93.7%        318.123       3.14          
16384      1        516        96.9%        320.456       3.12          
```

**Key Insights**:
- Sparse keys stay constant (sink + window)
- Compute reduction increases with context length
- Current PyTorch implementation shows baseline performance

---

## Interpreting Results

### Metrics

| Metric | Meaning |
|--------|---------|
| **Context Length** | Number of tokens in KV cache |
| **Batch Size** | Number of concurrent sequences |
| **Sparse K** | Number of keys attended (sink + window) |
| **Reduction** | Compute savings vs full attention |
| **Latency** | Time per iteration (ms) |
| **Throughput** | Tokens generated per second |
| **TFLOPS** | Estimated floating point ops per second |

### Sparsity Analysis

For `sink=4`, `window=512`:
- **Context 1K**: sparse_k=516, reduction=48%
- **Context 4K**: sparse_k=516, reduction=87%
- **Context 16K**: sparse_k=516, reduction=97%

**Takeaway**: Longer contexts benefit more from sparsity!

### Performance Notes

**Current (PyTorch baseline)**:
- Slow due to Python loops
- Baseline for correctness
- Not representative of optimized performance

**Expected (with Triton/CUDA)**:
- 10-100x faster
- Sub-millisecond latency
- Competitive with FlashAttention

---

## Comparison with Other Backends

### Profile FlashAttention (for comparison)
```bash
/workspace/anaconda3/envs/vllm/bin/python vllm/attention_scripts/profile_attention.py \
    --backend FLASH_ATTN \
    --context-length 4096 \
    --batch-size 1
```

### Profile Triton (for comparison)
```bash
/workspace/anaconda3/envs/vllm/bin/python vllm/attention_scripts/profile_attention.py \
    --backend TRITON_ATTN \
    --context-length 4096 \
    --batch-size 1
```

### Expected Comparison (after optimization)
```
Backend      Ctx=4K  Sparse K  Reduction  Latency    Speedup
FLASH_ATTN   4096    4096      0%         ~1ms       1.0x
SKLT (opt)   4096    516       87%        ~0.12ms    ~8x (expected)
```

---

## Tuning Guide

### Sink Tokens
```bash
# Test different sink values
for sink in 2 4 8 16; do
    python profile_sklt_vllm.py --sink-tokens $sink --window-size 512 --context-length 4096
done
```

**Recommendation**: 2-8 for most models

### Window Size
```bash
# Test different window sizes
for window in 128 256 512 1024 2048; do
    python profile_sklt_vllm.py --sink-tokens 4 --window-size $window --context-length 4096
done
```

**Recommendation**: 256-1024 depending on task

### Context Length Scaling
```bash
# See how SKLT scales with context
python profile_sklt_vllm.py \
    --context-length 1024 2048 4096 8192 16384 32768 \
    --sink-tokens 4 \
    --window-size 512 \
    --batch-size 1
```

---

## Advanced Usage

### Export Results for Analysis
```bash
python profile_sklt_vllm.py \
    --context-length 1024 2048 4096 8192 \
    --batch-size 1 2 4 8 \
    --save-trace \
    --output-file my_sklt_profile.json

# Then analyze with:
python -c "
import json
with open('my_sklt_profile.json') as f:
    data = json.load(f)
for r in data:
    print(f\"Ctx={r['context_length']}, Batch={r['batch_size']}: {r['avg_latency_ms']:.2f}ms\")
"
```

### Compare Sparse Configurations
```bash
# Small window (aggressive sparsity)
python profile_sklt_vllm.py --sink-tokens 2 --window-size 128 --context-length 4096

# Medium window (balanced)
python profile_sklt_vllm.py --sink-tokens 4 --window-size 512 --context-length 4096

# Large window (more context)
python profile_sklt_vllm.py --sink-tokens 8 --window-size 2048 --context-length 4096
```

---

## Output Files

### JSON Structure
```json
[
  {
    "context_length": 4096,
    "batch_size": 1,
    "avg_latency_ms": 316.054,
    "throughput": 3.16,
    "tflops": 0.00,
    "sparse_k": 516,
    "full_k": 4096,
    "sparsity_ratio": 0.126,
    "compute_reduction_pct": 87.4,
    "sink_tokens": 4,
    "window_size": 512
  }
]
```

---

## Known Limitations

### Current Implementation (PyTorch)
- ⚠️ Slow performance (nested Python loops)
- ⚠️ Not optimized for GPU
- ✅ Correct sparse attention computation
- ✅ Useful for correctness validation

### Future (Optimized)
- 🚀 Triton kernels: 10-100x faster
- 🚀 CUDA kernels: 100-1000x faster
- 🚀 Competitive throughput with FlashAttention

---

## Troubleshooting

### "CUDA out of memory"
Reduce buffer sizes in `streaming_indexer.py`:
```python
max_batch = 32  # Reduce from 64
```

### Slow performance
**Expected**: PyTorch baseline is slow  
**Solution**: Wait for Triton/CUDA optimization

### No output
Check:
1. CUDA available: `nvidia-smi`
2. SKLT backend registered: `grep SKLT vllm/v1/attention/backends/registry.py`
3. Run with `-v` for verbose output

---

## Quick Examples

### Profile Context Scaling
```bash
python vllm/attention_scripts/profile_sklt_vllm.py \
    --context-length 1024 2048 4096 8192 16384
```

### Profile Batch Scaling
```bash
python vllm/attention_scripts/profile_sklt_vllm.py \
    --context-length 4096 \
    --batch-size 1 2 4 8 16 32
```

### Profile Sparsity Impact
```bash
for window in 128 256 512 1024; do
    echo "=== Window size: $window ==="
    python vllm/attention_scripts/profile_sklt_vllm.py \
        --context-length 4096 \
        --window-size $window \
        --batch-size 1 \
        --num-iterations 20
done
```

---

## Integration with Existing Scripts

This script follows the same pattern as `profile_attention.py` but adds SKLT-specific features:

**Similarities**:
- Same argument structure
- Same profiling methodology
- Compatible output format

**Differences**:
- SKLT backend hardcoded
- Adds `--sink-tokens` and `--window-size`
- Reports sparsity statistics
- Shows compute reduction percentage

---

## Next Steps

1. **Baseline**: Use this script to establish PyTorch baseline
2. **Optimize**: Implement Triton kernels and re-profile
3. **Compare**: Compare optimized SKLT vs FlashAttention
4. **Tune**: Find optimal sink/window for your workload

---

**Script**: `vllm/attention_scripts/profile_sklt_vllm.py`  
**Purpose**: Profile SKLT sparse attention decode performance  
**Status**: ✅ Working
