# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Literal, Optional

from pydantic import field_validator
from pydantic.dataclasses import dataclass

from vllm.config.utils import config
from vllm.v1.attention.backends.registry import AttentionBackendEnum


@config
@dataclass
class IndexerConfig:
    """Configuration for sparse attention indexer."""
    
    indexer_type: str = "streaming"
    """Type of indexer: 'streaming', 'custom', etc."""
    
    # Streaming indexer specific parameters
    num_sink_tokens: int = 4
    """Number of initial tokens to always attend to (attention sink)."""
    
    local_window_size: int = 1024
    """Size of local attention window."""
    
    max_sparse_k: int = 2048
    """Maximum number of sparse keys per head (buffer allocation size)."""
    
    def __post_init__(self):
        """Validate configuration."""
        if self.num_sink_tokens < 0:
            raise ValueError(f"num_sink_tokens must be >= 0, got {self.num_sink_tokens}")
        if self.local_window_size < 0:
            raise ValueError(f"local_window_size must be >= 0, got {self.local_window_size}")
        if self.max_sparse_k < self.num_sink_tokens + self.local_window_size:
            raise ValueError(
                f"max_sparse_k ({self.max_sparse_k}) must be >= "
                f"num_sink_tokens + local_window_size ({self.num_sink_tokens + self.local_window_size})"
            )


@config
@dataclass
class AttentionConfig:
    """Configuration for attention mechanisms in vLLM."""

    backend: AttentionBackendEnum | None = None
    """Attention backend to use. If None, will be selected automatically."""

    flash_attn_version: Literal[2, 3] | None = None
    """Force vllm to use a specific flash-attention version (2 or 3).
    Only valid when using the flash-attention backend."""

    use_prefill_decode_attention: bool = False
    """Use separate prefill and decode kernels for attention instead of
    the unified triton kernel."""

    flash_attn_max_num_splits_for_cuda_graph: int = 32
    """Flash Attention max number splits for cuda graph decode."""

    use_cudnn_prefill: bool = False
    """Whether to use cudnn prefill."""

    use_trtllm_ragged_deepseek_prefill: bool = True
    """Whether to use TRTLLM ragged deepseek prefill."""

    use_trtllm_attention: bool | None = None
    """If set to True/False, use or don't use the TRTLLM attention backend
    in flashinfer. If None, auto-detect the attention backend in flashinfer."""

    disable_flashinfer_prefill: bool = False
    """Whether to disable flashinfer prefill."""

    disable_flashinfer_q_quantization: bool = False
    """If set, when using fp8 kv, do not quantize Q to fp8."""
    
    # SKLT sparse attention configuration
    indexer_config: Optional[IndexerConfig] = None
    """Configuration for sparse attention indexer (used by SKLT backend)."""
    
    use_sparse_attention: bool = False
    """Enable sparse attention patterns."""

    def compute_hash(self) -> str:
        """
        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        from vllm.config.utils import get_hash_factors, hash_factors

        ignored_factors: list[str] = []
        factors = get_hash_factors(self, ignored_factors)
        return hash_factors(factors)

    @field_validator("backend", mode="before")
    @classmethod
    def validate_backend_before(cls, value: Any) -> Any:
        """Enable parsing of the `backend` enum type from string."""
        if isinstance(value, str):
            return AttentionBackendEnum[value.upper()]
        return value
