# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SKLT attention implementation."""

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionImpl, AttentionType
from vllm.v1.attention.backends.sklt.ops import sklt_sparse_attention
from vllm.v1.attention.backends.sklt.sklt_metadata import SKLTAttentionMetadata

logger = init_logger(__name__)


class SKLTAttentionImpl(AttentionImpl[SKLTAttentionMetadata]):
    """SKLT sparse attention implementation.
    
    Uses sparse patterns from indexer to compute attention efficiently.
    """
    
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        """Initialize SKLT attention implementation.
        
        Args:
            num_heads: Number of query attention heads
            head_size: Dimension of each attention head
            scale: Attention scale factor (typically 1/sqrt(head_size))
            num_kv_heads: Number of KV heads (for GQA/MQA)
            alibi_slopes: ALiBi slopes (not supported yet)
            sliding_window: Sliding window size (handled by indexer)
            kv_cache_dtype: KV cache data type
            logits_soft_cap: Logits soft cap (not supported yet)
            attn_type: Type of attention (decoder, encoder, etc.)
            kv_sharing_target_layer_name: For cross-attention KV sharing
        """
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        
        # Validate configuration
        if alibi_slopes is not None:
            raise NotImplementedError("SKLT backend does not support ALiBi yet")
        
        if logits_soft_cap is not None:
            raise NotImplementedError("SKLT backend does not support logits soft cap yet")
        
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                f"SKLT backend only supports decoder attention, got {attn_type}"
            )
        
        # Store block size (will be set later from cache spec)
        self.block_size: int | None = None
        
        logger.info(
            f"SKLT: Initialized attention impl with "
            f"num_heads={num_heads}, head_size={head_size}, "
            f"num_kv_heads={self.num_kv_heads}, scale={self.scale}"
        )
    
    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: SKLTAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with SKLT sparse attention.
        
        Args:
            layer: Attention layer module
            query: Query tensor (num_tokens, num_heads, head_size)
            key: Key tensor (num_tokens, num_kv_heads, head_size)
            value: Value tensor (num_tokens, num_kv_heads, head_size)
            kv_cache: KV cache tensor (2, num_blocks, block_size, num_kv_heads, head_size)
            attn_metadata: SKLT attention metadata with sparsity info
            output: Pre-allocated output tensor
            output_scale: Output scale for quantization (not supported yet)
            output_block_scale: Output block scale for quantization (not supported yet)
            
        Returns:
            Attention output tensor (num_tokens, num_heads * head_size)
        """
        if output is None:
            num_tokens = query.shape[0]
            output = torch.empty(
                (num_tokens, self.num_heads, self.head_size),
                dtype=query.dtype,
                device=query.device
            )
        
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "SKLT backend does not support fused output quantization yet"
            )
        
        if attn_metadata is None:
            # Profiling run
            return output.view(output.shape[0], -1)
        
        # Extract key and value caches
        key_cache, value_cache = kv_cache.unbind(0)
        
        # Infer block size from cache shape if not already set
        if self.block_size is None:
            self.block_size = key_cache.shape[1]  # (num_blocks, block_size, ...)
            logger.info(f"SKLT: Inferred block_size={self.block_size} from KV cache")
        
        num_actual_tokens = attn_metadata.num_actual_tokens
        
        # SKLT is optimized for decode (single query token per sequence)
        # For prefill (max_query_len > 1), fallback to standard attention
        if attn_metadata.max_query_len > 1:
            logger.warning_once(
                f"SKLT backend: Prefill detected (max_query_len={attn_metadata.max_query_len}). "
                "Falling back to standard FlashAttention-style computation. "
                "For optimal performance, use FLASH_ATTN backend for prefill phase.",
                scope="sklt_prefill_fallback"
            )
            logger.info(f"SKLT: Prefill")
            # Fallback to standard attention for prefill
            return self._forward_prefill_fallback(
                query=query,
                key=key,
                value=value,
                key_cache=key_cache,
                value_cache=value_cache,
                output=output,
                attn_metadata=attn_metadata,
                num_actual_tokens=num_actual_tokens,
            )
        logger.info(f"SKLT: Decoding")

        # Decode path: Use SKLT sparse attention
        sklt_sparse_attention(
            query=query[:num_actual_tokens],
            key_cache=key_cache,
            value_cache=value_cache,
            sparsity_info=attn_metadata.sparsity_info,
            block_table=attn_metadata.block_table,
            query_start_loc=attn_metadata.query_start_loc,
            seq_lens=attn_metadata.seq_lens,
            block_size=self.block_size,
            scale=self.scale,
            num_kv_heads=self.num_kv_heads,
            output=output[:num_actual_tokens],
        )
        
        # Reshape output to (num_tokens, num_heads * head_size)
        return output.view(output.shape[0], -1)
    
    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Update KV cache with new key/value tensors.
        
        This is called separately from forward() when forward_includes_kv_cache_update=False.
        We use the standard reshape_and_cache operation from FlashAttention.
        """
        # Skip if no keys/values to cache
        if key is None or value is None:
            return
        
        # Skip if sharing KV cache with another layer
        if self.kv_sharing_target_layer_name is not None:
            return
        
        # Import reshape_and_cache from FlashAttention utils
        try:
            from vllm.v1.attention.backends.fa_utils import reshape_and_cache_flash
        except ImportError:
            # Fallback to manual caching if reshape_and_cache not available
            logger.warning_once(
                "reshape_and_cache_flash not available, using manual KV cache update",
                scope="sklt_kv_cache_update"
            )
            self._manual_kv_cache_update(key, value, kv_cache, slot_mapping)
            return
        
        key_cache, value_cache = kv_cache.unbind(0)
        
        # Use the standard reshape_and_cache operation
        reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )
    
    def _manual_kv_cache_update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Manual KV cache update fallback."""
        key_cache, value_cache = kv_cache.unbind(0)
        
        # Simple scatter operation
        num_tokens = slot_mapping.shape[0]
        for i in range(num_tokens):
            slot = slot_mapping[i].item()
            if slot < 0:
                continue
            
            # Convert slot to block coordinates
            block_idx = slot // self.block_size
            block_offset = slot % self.block_size
            
            # Update cache
            key_cache[block_idx, block_offset] = key[i]
            value_cache[block_idx, block_offset] = value[i]
    
    def _forward_prefill_fallback(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: SKLTAttentionMetadata,
        num_actual_tokens: int,
    ) -> torch.Tensor:
        """Fallback to standard attention for prefill phase.
        
        Uses PyTorch's scaled_dot_product_attention for correctness.
        This is not optimized but allows SKLT to handle prefill.
        """
        batch_size = attn_metadata.query_start_loc.shape[0] - 1
        query_start_loc_cpu = attn_metadata.query_start_loc.cpu()
        seq_lens_cpu = attn_metadata.seq_lens.cpu()
        block_table_cpu = attn_metadata.block_table.cpu()
        
        num_heads_per_kv = self.num_heads // self.num_kv_heads
        
        # Process each sequence
        for b in range(batch_size):
            q_start = query_start_loc_cpu[b].item()
            q_end = query_start_loc_cpu[b + 1].item()
            num_q = q_end - q_start
            
            if num_q == 0:
                continue
            
            seq_len = seq_lens_cpu[b].item()
            
            # Get query for this sequence
            q_seq = query[q_start:q_end]  # (num_q, num_heads, head_size)
            
            # Gather full KV cache for this sequence
            k_list = []
            v_list = []
            
            for pos in range(seq_len):
                block_idx = pos // self.block_size
                block_offset = pos % self.block_size
                physical_block = block_table_cpu[b, block_idx].item()
                
                # For each KV head
                for kv_h in range(self.num_kv_heads):
                    k_vec = key_cache[physical_block, block_offset, kv_h]
                    v_vec = value_cache[physical_block, block_offset, kv_h]
                    k_list.append(k_vec)
                    v_list.append(v_vec)
            
            # Stack into tensors: (seq_len, num_kv_heads, head_size)
            k_seq = torch.stack(k_list).view(seq_len, self.num_kv_heads, self.head_size)
            v_seq = torch.stack(v_list).view(seq_len, self.num_kv_heads, self.head_size)
            
            # Expand KV for GQA: (seq_len, num_heads, head_size)
            if num_heads_per_kv > 1:
                k_seq = k_seq.repeat_interleave(num_heads_per_kv, dim=1)
                v_seq = v_seq.repeat_interleave(num_heads_per_kv, dim=1)
            
            # Transpose for SDPA: (num_heads, num_q, head_size) and (num_heads, seq_len, head_size)
            q_sdpa = q_seq.transpose(0, 1)  # (num_heads, num_q, head_size)
            k_sdpa = k_seq.transpose(0, 1)  # (num_heads, seq_len, head_size)
            v_sdpa = v_seq.transpose(0, 1)  # (num_heads, seq_len, head_size)
            
            # Apply causal mask for prefill
            attn_mask = None
            if attn_metadata.causal:
                # Create causal mask: query positions can only attend to earlier key positions
                # Query positions are at [seq_len - num_q : seq_len]
                ctx_len = seq_len - num_q
                # Create causal mask
                attn_mask = torch.ones(num_q, seq_len, dtype=torch.bool, device=query.device)
                for i in range(num_q):
                    q_pos = ctx_len + i
                    attn_mask[i, :q_pos+1] = False  # Can attend to positions up to q_pos
            
            # Compute attention using PyTorch SDPA
            out_seq = F.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                attn_mask=attn_mask,
                scale=self.scale,
            )  # (num_heads, num_q, head_size)
            
            # Transpose back: (num_q, num_heads, head_size)
            out_seq = out_seq.transpose(0, 1)
            
            # Write to output
            output[q_start:q_end] = out_seq
        
        # Reshape output to (num_tokens, num_heads * head_size)
        return output.view(output.shape[0], -1)
