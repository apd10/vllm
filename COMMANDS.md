# SKLT Backend - Command Reference

## 🎉 Quick Start - Run SKLT Backend NOW!

### Full Test Suite (Recommended)
```bash
bash /workspace/vllm/RUN_SKLT.sh
```
**Result**: Runs all 3 tests, shows comprehensive validation ✅

---

## Individual Test Commands

### 1. Component Test
Tests: Imports, config, indexer, kernel
```bash
/workspace/anaconda3/envs/vllm/bin/python /workspace/vllm/test_sklt_simple.py
```

### 2. Decode-Only Test
Tests: SKLT sparse attention in pure decode scenario
```bash
/workspace/anaconda3/envs/vllm/bin/python /workspace/vllm/test_sklt_decode_only.py
```

### 3. Prefill + Decode Test
Tests: Both prefill fallback and decode sparse attention
```bash
/workspace/anaconda3/envs/vllm/bin/python /workspace/vllm/test_sklt_prefill_fallback.py
```

### 4. Full LLM Example ⭐
Tests: Complete text generation with vLLM
```bash
/workspace/anaconda3/envs/vllm/bin/python /workspace/vllm/examples/sklt_example.py
```
**Result**: Generates text using SKLT backend! ✅

---

## Python API Usage

### Minimal Example
```python
from vllm import LLM, SamplingParams
from vllm.config.attention import AttentionConfig, IndexerConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum

# Configure SKLT
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

# Create LLM
llm = LLM(
    model="facebook/opt-125m",
    attention_config=attention_config,
    enforce_eager=True,
)

# Generate
prompts = ["Hello, my name is"]
params = SamplingParams(temperature=0.8, max_tokens=50)
outputs = llm.generate(prompts, params)
print(outputs[0].outputs[0].text)
```

### Save as Script and Run
```bash
# Save the above to my_sklt_test.py, then:
/workspace/anaconda3/envs/vllm/bin/python my_sklt_test.py
```

---

## Command Line Interface (vllm serve)

### Basic Server
```bash
/workspace/anaconda3/envs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
    --model facebook/opt-125m \
    --attention-backend SKLT \
    --attention-config '{"indexer_config": {"indexer_type": "streaming", "num_sink_tokens": 4, "local_window_size": 512, "max_sparse_k": 1024}, "use_sparse_attention": true}' \
    --enforce-eager \
    --port 8000
```

### Using Structured Config Format
```bash
/workspace/anaconda3/envs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
    --model facebook/opt-125m \
    -ac.backend SKLT \
    -ac.use_sparse_attention true \
    -ac.indexer_config.indexer_type streaming \
    -ac.indexer_config.num_sink_tokens 4 \
    -ac.indexer_config.local_window_size 512 \
    -ac.indexer_config.max_sparse_k 1024 \
    --enforce-eager \
    --port 8000
```

---

## Configuration Parameters

### IndexerConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `indexer_type` | str | "streaming" | Type of sparse pattern indexer |
| `num_sink_tokens` | int | 4 | Number of initial tokens to always attend to |
| `local_window_size` | int | 1024 | Size of local attention window |
| `max_sparse_k` | int | 2048 | Maximum sparse keys per head (buffer size) |

### AttentionConfig (SKLT-specific)

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `backend` | AttentionBackendEnum | ✅ | Set to `AttentionBackendEnum.SKLT` |
| `indexer_config` | IndexerConfig | ✅ | Sparse pattern configuration |
| `use_sparse_attention` | bool | ✅ | Must be `True` for SKLT |

---

## Environment

### Python
```bash
/workspace/anaconda3/envs/vllm/bin/python
```

### Conda Environment
```bash
# Activate (if needed)
source /workspace/anaconda3/bin/activate
conda activate vllm
```

---

## Troubleshooting

### "CUDA out of memory" during initialization
**Solution**: Reduce buffer sizes in `streaming_indexer.py` line 32:
```python
max_batch = 32  # Reduce from 64
```

### "use_sparse=True required"
**Solution**: Add to config:
```python
attention_config = AttentionConfig(
    ...
    use_sparse_attention=True,  # Add this
)
```

### Prefill is slow
**Expected**: Prefill uses fallback (not optimized)  
**Future**: Will be optimized with sparse prefill kernels

---

## Quick Reference

| What | Command |
|------|---------|
| **Run All Tests** | `bash /workspace/vllm/RUN_SKLT.sh` |
| **Component Test** | `/workspace/anaconda3/envs/vllm/bin/python test_sklt_simple.py` |
| **Decode Test** | `/workspace/anaconda3/envs/vllm/bin/python test_sklt_decode_only.py` |
| **Prefill+Decode** | `/workspace/anaconda3/envs/vllm/bin/python test_sklt_prefill_fallback.py` |
| **Full Example** | `/workspace/anaconda3/envs/vllm/bin/python examples/sklt_example.py` |

---

## Documentation

| Document | Purpose |
|----------|---------|
| `SKLT_FINAL_SUMMARY.md` | Quick overview & status |
| `SKLT_QUICKSTART.md` | Getting started guide |
| `SKLT_USAGE.md` | Comprehensive usage documentation |
| `IMPLEMENTATION_SUMMARY.md` | Technical implementation details |
| `PLAN.md` | Original implementation plan |
| `COMMANDS.md` | This file - command reference |

---

**Status**: ✅ FULLY WORKING  
**Test Command**: `bash /workspace/vllm/RUN_SKLT.sh`  
**Example Command**: `/workspace/anaconda3/envs/vllm/bin/python examples/sklt_example.py`
