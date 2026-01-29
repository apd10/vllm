# Attention Backend Profiling Script

This script profiles the **DECODE** scenario: measuring the time to generate one new token when the KV cache already contains `context_length` tokens.

## profile_attention.py

A profiling script for attention backend decode performance that measures latency, throughput, and TFLOPS.

### What This Script Measures

**DECODE Mode (Token Generation):**
- Query length = 1 (generating 1 new token)
- KV cache size = context_length (already computed tokens)
- Measures: Time to attend over `context_length` cached tokens to generate the next token

This is the most common scenario during inference/serving.

### Features

- Profile any attention backend (FlashAttention, FlashInfer, Triton, etc.)
- Test multiple context lengths and batch sizes
- Measure latency, throughput, and TFLOPS
- Optional detailed profiling with PyTorch profiler
- Support for different data types (float16, bfloat16)
- Configurable model parameters (heads, head size, block size)

### Requirements

- CUDA-capable GPU
- PyTorch with CUDA support
- vLLM installed: `conda activate vllm` or `pip install -e .`

### Usage

#### Basic Usage

Profile FlashAttention with default settings:
```bash
python vllm/attention_scripts/profile_attention.py --backend FLASH_ATTN
```

#### Profile Specific Context Lengths

```bash
python vllm/attention_scripts/profile_attention.py \
    --backend FLASH_ATTN \
    --context-length 1024 2048 4096 8192
```

#### Profile Multiple Batch Sizes

```bash
python vllm/attention_scripts/profile_attention.py \
    --backend FLASH_ATTN \
    --context-length 4096 \
    --batch-size 1 4 8 16 32
```

#### Use PyTorch Profiler for Detailed Analysis

```bash
python vllm/attention_scripts/profile_attention.py \
    --backend FLASH_ATTN \
    --context-length 4096 \
    --batch-size 8 \
    --use-profiler
```

This will generate a Chrome trace file that can be viewed in `chrome://tracing`.

#### Configure Model Parameters (e.g., for GQA)

```bash
python vllm/attention_scripts/profile_attention.py \
    --backend FLASH_ATTN \
    --context-length 4096 \
    --num-heads 32 \
    --num-kv-heads 8 \
    --head-size 128 \
    --block-size 16 \
    --dtype bfloat16
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--backend` | str | `FLASH_ATTN` | Attention backend to profile |
| `--context-length` | int+ | `[1024, 2048, 4096]` | Context lengths to profile |
| `--batch-size` | int+ | `[1, 4, 8]` | Batch sizes to profile |
| `--num-warmup` | int | `10` | Number of warmup iterations |
| `--num-iterations` | int | `100` | Number of benchmark iterations |
| `--use-profiler` | flag | `False` | Use torch profiler for detailed analysis |
| `--dtype` | str | `float16` | Data type (`float16` or `bfloat16`) |
| `--num-heads` | int | `32` | Number of attention heads |
| `--num-kv-heads` | int | `32` | Number of KV heads (for GQA) |
| `--head-size` | int | `128` | Size of each attention head |
| `--block-size` | int | `16` | KV cache block size |

### Available Backends

Common backends:
- `FLASH_ATTN` - FlashAttention (FA2/FA3)
- `FLASHINFER` - FlashInfer
- `TRITON_ATTN` - Triton Attention
- `FLEX_ATTENTION` - PyTorch Flex Attention
- `TREE_ATTN` - Tree Attention

See `--help` for the complete list of available backends.

### Output

The script provides:

1. **Setup Information**: Tensor shapes, memory allocation
2. **Per-Configuration Results**: Latency, throughput, TFLOPS for each (context_length, batch_size) pair
3. **Summary Table**: Comparison across all tested configurations
4. **Optional Profiler Output**: Detailed kernel-level timing information

Example output:
```
================================================================================
Profiling FLASH_ATTN with context_length=4096, batch_size=8
================================================================================

Setup complete:
  - Query shape: torch.Size([8, 32, 128])
  - Key shape: torch.Size([8, 32, 128])
  - Value shape: torch.Size([8, 32, 128])
  - KV cache shape: torch.Size([2, 2148, 16, 32, 128])
  - Output shape: torch.Size([8, 4096])
  - Num tokens: 8
  - Num blocks: 2148

Warming up (10 iterations)...
Benchmarking (100 iterations)...

================================================================================
Results:
================================================================================
Average GPU latency: 0.523 ms
Average CPU latency: 0.531 ms
Throughput: 15296.37 sequences/sec
Estimated TFLOPS: 12.45
Memory bandwidth (approx): 1234.56 GB/s
================================================================================

================================================================================
SUMMARY
================================================================================
Context Length  Batch Size   Latency (ms)    Throughput      TFLOPS    
--------------------------------------------------------------------------------
4096            8            0.523           15296.37        12.45     
================================================================================
```

### Notes

- The script profiles **DECODE** scenarios (query_length=1), measuring token generation time
- `context_length` parameter = number of tokens already in the KV cache
- The attention operation attends over all `context_length` cached tokens to generate 1 new token
- TFLOPS calculations are approximate and based on standard attention FLOP counts
- For accurate profiling, ensure the GPU is not under load from other processes
- Use `--use-profiler` sparingly as it adds overhead and generates large trace files

### Understanding the Output

When you run with `--context-length 4096 --batch-size 8`:
- Each of the 8 sequences has 4096 tokens in its KV cache
- The script measures how long it takes to generate 1 new token for each sequence
- The latency shown is the time for the attention operation over the 4096-token context

### Troubleshooting

**FLASH_ATTN PTX Compatibility Error on B200/Blackwell GPUs**:
```
ERROR: CUDA PTX Compatibility Issue
Backend FLASH_ATTN is not compatible with your GPU/CUDA setup.
```
**Solution**: Use TRITON_ATTN instead:
```bash
python vllm/attention_scripts/profile_attention.py --backend TRITON_ATTN --context-length 4096
```

This happens when FlashAttention was compiled for older GPU architectures and your newer GPU (like B200 with compute capability 10.0) requires recompilation.

**Backend not available**: Some backends require specific hardware or additional dependencies.
```bash
# For FlashAttention
pip install flash-attn --no-build-isolation

# For FlashInfer
pip install flashinfer
```

**Out of memory**: Reduce batch size or context length.

**Slow profiling**: Reduce `--num-iterations` for faster results (at the cost of less accurate measurements).

### Example Workflows

#### Find Optimal Backend for Your Hardware
```bash
# Test FlashAttention
python vllm/attention_scripts/profile_attention.py --backend FLASH_ATTN --context-length 4096

# Test FlashInfer
python vllm/attention_scripts/profile_attention.py --backend FLASHINFER --context-length 4096

# Compare results and pick the fastest
```

#### Test Long Context Performance
```bash
python vllm/attention_scripts/profile_attention.py \
    --backend FLASH_ATTN \
    --context-length 8192 16384 32768 \
    --batch-size 1
```

#### Tune Batch Size
```bash
python vllm/attention_scripts/profile_attention.py \
    --backend FLASH_ATTN \
    --context-length 4096 \
    --batch-size 1 2 4 8 16 32
```

#### Debug Performance Issues
```bash
python vllm/attention_scripts/profile_attention.py \
    --backend FLASH_ATTN \
    --context-length 4096 \
    --use-profiler \
    --num-iterations 10
```

Then open the generated trace file in Chrome at `chrome://tracing`.
