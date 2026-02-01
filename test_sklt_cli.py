#!/usr/bin/env python3
"""
Minimal CLI test for SKLT backend with vLLM.
"""

import sys

def main():
    print("=" * 70)
    print("SKLT Backend CLI Test")
    print("=" * 70)
    
    try:
        from vllm import LLM, SamplingParams
        from vllm.config.attention import AttentionConfig, IndexerConfig
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
        print("✓ Imports successful")
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    try:
        # Configure SKLT backend
        indexer_config = IndexerConfig(
            indexer_type="streaming",
            num_sink_tokens=4,
            local_window_size=256,
            max_sparse_k=512,
        )
        
        attention_config = AttentionConfig(
            backend=AttentionBackendEnum.SKLT,
            indexer_config=indexer_config,
            use_sparse_attention=True,
        )
        
        print(f"✓ Configuration created")
        print(f"  - Backend: {attention_config.backend.name}")
        print(f"  - Indexer: {indexer_config.indexer_type}")
        print(f"  - Sink tokens: {indexer_config.num_sink_tokens}")
        print(f"  - Window size: {indexer_config.local_window_size}")
        
    except Exception as e:
        print(f"✗ Configuration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    try:
        print("\nInitializing LLM (this may take a moment)...")
        print("  Note: SKLT is decode-only. Prefill will use default backend.")
        
        # IMPORTANT: SKLT backend is decode-only
        # For a production setup, you would:
        # 1. Use chunked prefill with a different backend for prefill phase
        # 2. Switch to SKLT for decode phase
        # For this test, we'll use a small prompt to minimize prefill
        
        llm = LLM(
            model="facebook/opt-125m",
            attention_config=attention_config,
            max_model_len=512,
            enforce_eager=True,
            gpu_memory_utilization=0.5,  # Reduce memory usage
        )
        print("✓ LLM initialized with SKLT backend")
        
    except Exception as e:
        print(f"✗ LLM initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    try:
        # Test generation
        print("\nTesting text generation...")
        prompts = ["Hello, my name is"]
        sampling_params = SamplingParams(
            temperature=0.8,
            max_tokens=20,
        )
        
        outputs = llm.generate(prompts, sampling_params)
        
        print("✓ Generation successful")
        print(f"\nPrompt: {prompts[0]}")
        print(f"Output: {outputs[0].outputs[0].text}")
        
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "=" * 70)
    print("✅ SKLT backend test PASSED!")
    print("=" * 70)
    return 0

if __name__ == "__main__":
    sys.exit(main())
