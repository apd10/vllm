# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time

from vllm import LLM, SamplingParams

# Sample prompts.
# Create a sampling params object.
sampling_params = SamplingParams(max_tokens=4, temperature=0.8, top_p=0.95)


def get_prompts(token_length, tokenizer):
    # create a random text of token_length tokens wr.t the tokenizer
    if token_length <= 0:
        return ""

    base_text = (
        "Hello, my name is John Doe. I am a software engineer. "
        "I live in New York City. "
    )

    def encode(text):
        if hasattr(tokenizer, "encode"):
            try:
                return tokenizer.encode(text, add_special_tokens=False)
            except TypeError:
                return tokenizer.encode(text)
        encoded = tokenizer(text, add_special_tokens=False)
        return getattr(encoded, "input_ids", encoded)

    def decode(token_ids):
        if hasattr(tokenizer, "decode"):
            try:
                return tokenizer.decode(token_ids, skip_special_tokens=True)
            except TypeError:
                return tokenizer.decode(token_ids)
        return tokenizer.decode(token_ids)

    base_tokens = encode(base_text)
    if not base_tokens:
        return base_text

    tokens = []
    while len(tokens) < token_length:
        tokens.extend(base_tokens)

    tokens = tokens[:token_length]
    return decode(tokens)

def main():
    # Create an LLM.
    llm = LLM(
        model="Qwen/Qwen3-4B-Instruct-2507",
        tensor_parallel_size=1,
        profiler_config={
            "profiler": "torch",
            "torch_profiler_dir": "./vllm_profile",
        },
        max_model_len=128000,
    )
    # get prompt
    prompts = [get_prompts(128000, llm.get_tokenizer())]
    # warmup?
    print("Length of prompt: ", len(prompts[0]))
    for _ in range(2):
        llm.generate(prompts, sampling_params)

    llm.start_profile()
    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    llm.stop_profile()

    # Print the outputs.
    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)

    # Add a buffer to wait for profiler in the background process
    # (in case MP is on) to finish writing profiling output.
    time.sleep(10)


if __name__ == "__main__":
    main()

