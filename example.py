from vllm import LLM, SamplingParams
from vllm.config import AttentionConfig

# -----------------------------
# Configuration
# -----------------------------
# Note: `Qwen/Qwen3.5-4B` does not exist on HF Hub. Using the cached
# `Qwen/Qwen3-4B-Instruct-2507` (H_q=32, H_kv=8, head_dim=128), which
# satisfies the FLASHINFER_SPARSE constraints
# (group_size in {1,2,4,8,16}, head_dim in {64,128,256}).
model_name = "Qwen/Qwen3-4B-Instruct-2507"


def main() -> None:
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=256,
    )

    # -----------------------------
    # Initialize LLM with FlashInfer-Sparse
    # -----------------------------
    # `channel_num=-1` means the score kernel uses the full head_dim for
    # the top-k selection (no partial-D approximation). `topk=128` keeps
    # the 128 most relevant KV tokens per (batch, query head) at decode.
    llm = LLM(
        model=model_name,
        dtype="bfloat16",                 # or "float16" depending on GPU
        gpu_memory_utilization=0.5,
        max_model_len=8192,               # smoke test, full ctx needs ~36 GiB
        trust_remote_code=True,
        attention_config=AttentionConfig(
            backend="FLASHINFER_SPARSE",
            topk=128,
            channel_num=-1,
        ),
    )

    prompts = [
        "Explain the difference between attention and linear attention.",
        "Write a short poem about Mumbai at night.",
    ]

    outputs = llm.generate(prompts, sampling_params)

    for i, output in enumerate(outputs):
        print(f"\n=== Prompt {i} ===")
        print(prompts[i])
        print("\n--- Generated ---")
        print(output.outputs[0].text)


if __name__ == "__main__":
    main()
