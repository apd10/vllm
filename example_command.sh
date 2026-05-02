vllm serve Qwen/Qwen3-4B-Instruct-2507 --attention-config.backend=FLASHINFER_SPARSE --attention-config.topk=0.05 --attention-config.channel_num=-1 --max-model-len 32000
