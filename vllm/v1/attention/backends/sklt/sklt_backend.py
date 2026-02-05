# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SKLT (SkyLight) sparse attention backend."""

from typing import ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import AttentionBackend, AttentionType, MultipleOf
from vllm.v1.attention.backends.sklt.sklt_impl import SKLTAttentionImpl
from vllm.v1.attention.backends.sklt.sklt_metadata import (
    SKLTAttentionMetadataBuilder,
)

logger = init_logger(__name__)


class SKLTAttentionBackend(AttentionBackend):
    """SKLT (SkyLight) sparse attention backend.
    
    Supports arbitrary per-head sparse masks with configurable indexers.
    Initial implementation uses PyTorch for correctness, optimized kernels
    will be added later.
    """
    
    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
    ]
    forward_includes_kv_cache_update: bool = False
    
    @staticmethod
    def get_name() -> str:
        """Return backend name."""
        return "SKLT"
    
    @staticmethod
    def get_impl_cls() -> type[SKLTAttentionImpl]:
        """Return implementation class."""
        return SKLTAttentionImpl
    
    @staticmethod
    def get_builder_cls() -> type[SKLTAttentionMetadataBuilder]:
        """Return metadata builder class."""
        return SKLTAttentionMetadataBuilder
    
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        """Get KV cache tensor shape.
        
        Uses standard FlashAttention-style layout:
        (2, num_blocks, block_size, num_kv_heads, head_size)
        
        where the first dimension separates key (0) and value (1) caches.
        """
        return (2, num_blocks, block_size, num_kv_heads, head_size)
    
    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        """Return supported block sizes.
        
        SKLT supports common block sizes used in vLLM.
        """
        return [16, 32, 64, 128]
    
    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        """Check if head size is supported.
        
        SKLT supports common head sizes.
        """
        return head_size in [64, 80, 96, 128, 256]
    
    @classmethod
    def supports_dtype(cls, dtype: torch.dtype) -> bool:
        """Check if dtype is supported."""
        return dtype in cls.supported_dtypes
    
    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        """Check if KV cache dtype is supported."""
        if kv_cache_dtype is None:
            return True
        return kv_cache_dtype in cls.supported_kv_cache_dtypes
    
    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        """Check if compute capability is supported.
        
        SKLT PyTorch implementation works on any CUDA device.
        For optimized kernels (future), may require higher compute capability.
        """
        # For PyTorch implementation, require at least SM 7.0 (Volta)
        return capability >= DeviceCapability(7, 0)
    
    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """Check if attention type is supported.
        
        Currently only supports decoder attention.
        """
        return attn_type == AttentionType.DECODER
    
    @classmethod
    def is_sparse(cls) -> bool:
        """SKLT is a sparse attention backend."""
        return True
    
    @classmethod
    def supports_sink(cls) -> bool:
        """SKLT supports sink tokens (via streaming indexer)."""
        return True
    
    @classmethod
    def supports_mm_prefix(cls) -> bool:
        """SKLT does not support multimodal prefix yet."""
        return False
    
    @classmethod
    def is_mla(cls) -> bool:
        """SKLT is not an MLA backend."""
        return False
    
    @classmethod
    def validate_configuration(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
        attn_type: str,
    ) -> list[str]:
        """Validate configuration for SKLT backend.
        
        Returns:
            List of invalid reasons (empty if valid)
        """
        invalid_reasons = super().validate_configuration(
            head_size=head_size,
            dtype=dtype,
            kv_cache_dtype=kv_cache_dtype,
            block_size=block_size,
            use_mla=use_mla,
            has_sink=has_sink,
            use_sparse=use_sparse,
            use_mm_prefix=use_mm_prefix,
            device_capability=device_capability,
            attn_type=attn_type,
        )
        
        # Additional SKLT-specific validation
        # SKLT requires sparse attention to be enabled
        if not use_sparse:
            invalid_reasons.append(
                "SKLT backend requires use_sparse=True. "
                "Set attention_config.use_sklt_sparse_attention=True"
            )
        
        return invalid_reasons
    
    @classmethod
    def supports_prefill(cls) -> bool:
        """SKLT supports prefill via fallback to standard attention.
        
        The implementation:
        - Decode (max_query_len=1): Uses SKLT sparse attention (optimized)
        - Prefill (max_query_len>1): Falls back to standard attention
        
        For production, consider using a dedicated prefill backend.
        """
        return True  # Via fallback to standard attention
