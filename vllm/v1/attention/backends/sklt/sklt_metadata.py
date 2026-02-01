# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metadata and builder for SKLT sparse attention."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.sklt.indexer import (
    BaseIndexer,
    SparsityInfo,
    StreamingIndexer,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


@dataclass
class SKLTAttentionMetadata:
    """Metadata for SKLT sparse attention."""
    
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    
    # SKLT-specific: sparsity information
    sparsity_info: SparsityInfo
    
    # Optional cascade attention (future)
    use_cascade: bool = False
    common_prefix_len: int = 0
    
    causal: bool = True


class SKLTAttentionMetadataBuilder(AttentionMetadataBuilder[SKLTAttentionMetadata]):
    """Metadata builder for SKLT sparse attention."""
    
    # SKLT currently uses PyTorch implementation with CPU synchronization (.cpu(), .item())
    # which is not compatible with CUDA graph capture. Marking as NEVER until we have
    # a CUDA-native kernel implementation.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER
    
    def __init__(
        self,
        kv_cache_spec: "AttentionSpec",
        layer_names: list[str],
        vllm_config: "VllmConfig",
        device: torch.device,
    ):
        """Initialize metadata builder.
        
        Args:
            kv_cache_spec: KV cache specification
            layer_names: Names of attention layers
            vllm_config: vLLM configuration
            device: Device to build metadata on
        """
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        
        # Initialize indexer from config
        indexer_config = vllm_config.attention_config.indexer_config
        if indexer_config is None:
            raise ValueError(
                "SKLT backend requires indexer_config in AttentionConfig. "
                "Please set attention_config.indexer_config."
            )
        
        # Factory pattern for different indexers
        if indexer_config.indexer_type == "streaming":
            self.indexer: BaseIndexer = StreamingIndexer(indexer_config, device)
            logger.info(
                f"SKLT: Initialized StreamingIndexer with "
                f"num_sink_tokens={indexer_config.num_sink_tokens}, "
                f"local_window_size={indexer_config.local_window_size}"
            )
        else:
            raise ValueError(
                f"Unknown indexer type: {indexer_config.indexer_type}. "
                f"Supported types: ['streaming']"
            )
        
        # Store model configuration
        self.num_heads_q = vllm_config.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        self.num_heads_kv = vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config
        )
        
        logger.info(
            f"SKLT: MetadataBuilder initialized with "
            f"num_query_heads={self.num_heads_q}, num_kv_heads={self.num_heads_kv}"
        )
    
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> SKLTAttentionMetadata:
        """Build SKLT attention metadata.
        
        Args:
            common_prefix_len: Length of common prefix (for cascade attention)
            common_attn_metadata: Common attention metadata
            fast_build: Whether to skip expensive computations
            
        Returns:
            SKLTAttentionMetadata with sparsity information
        """
        # Extract common metadata
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        causal = common_attn_metadata.causal
        
        # Compute sparsity information ONLY for decode (max_query_len = 1)
        # For prefill, we'll use fallback attention and sparsity_info won't be used
        sparsity_info = None
        
        if max_query_len == 1:
            # Decode phase: compute sparse patterns
            sparsity_info = self.indexer.compute_sparsity(
                batch_size=num_reqs,
                num_queries=num_actual_tokens,
                num_query_heads=self.num_heads_q,
                num_kv_heads=self.num_heads_kv,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
                attn_metadata=common_attn_metadata,
            )
            
            # Validate sparsity info
            try:
                sparsity_info.validate_shapes()
            except AssertionError as e:
                logger.error(f"SKLT: Sparsity info validation failed: {e}")
                raise
        else:
            # Prefill phase: create dummy sparsity info (won't be used)
            # This is just to satisfy the metadata structure
            logger.debug(
                f"SKLT: Prefill detected (max_query_len={max_query_len}). "
                "Skipping sparsity computation - will use fallback attention."
            )
            # Create empty/dummy sparsity info
            from vllm.v1.attention.backends.sklt.indexer import SparsityInfo
            sparsity_info = SparsityInfo(
                sparse_len=torch.zeros((num_reqs, 1, self.num_heads_q, 1), 
                                      dtype=torch.int32, device=self.device),
                sparse_idx=torch.zeros((num_reqs, 1, self.num_heads_q, 1), 
                                      dtype=torch.int32, device=self.device),
                sparse_weights=torch.zeros((num_reqs, 1, self.num_heads_q, 1), 
                                          dtype=torch.float32, device=self.device),
            )
        
        return SKLTAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            sparsity_info=sparsity_info,
            use_cascade=common_prefix_len > 0,
            common_prefix_len=common_prefix_len,
            causal=causal,
        )
