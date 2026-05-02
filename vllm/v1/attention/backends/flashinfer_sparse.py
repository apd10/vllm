# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer attention backend with oracle top-k sparse decode.

Same as ``FLASHINFER`` for prefill, cascade attention and the KV-cache
update path. The per-step decode wrapper is swapped for the standalone
oracle top-k sparse wrapper from the ``sparse_oracle_topk_optimized``
package, which:

  1. computes dense ``Q @ K^T`` selection scores using only the first
     ``channel_num`` head channels (``-1`` means full ``head_dim``),
  2. picks the ``topk`` largest indices per ``(batch, query_head)``,
  3. runs paged sparse decode over the selected indices using the full
     ``head_dim``.

The ``topk`` and ``channel_num`` knobs are read from
``AttentionConfig`` and may be supplied via ``--attention-config`` on the
CLI or by passing ``attention_config={...}`` to ``LLM(...)``.
"""

from typing import Any, ClassVar

import torch
from typing_extensions import override

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import AttentionCGSupport, MultipleOf
from vllm.v1.attention.backends.flashinfer import (
    FlashInferBackend,
    FlashInferImpl,
    FlashInferMetadataBuilder,
)
from vllm.v1.attention.backends.utils import get_kv_cache_layout
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


def _import_sparse_oracle_topk_wrapper_cls():
    """Lazily import the standalone sparse oracle top-k wrapper.

    Kept lazy so that ``import vllm`` does not require the optional
    ``sparse_oracle_topk_optimized`` package to be installed.
    """
    try:
        from sparse_oracle_topk_optimized import (
            BatchDecodeWithPagedKVCacheWrapper as SparseOracleTopKDecodeWrapper,
        )
    except ImportError as e:
        raise ImportError(
            "FLASHINFER_SPARSE backend requires the standalone "
            "`sparse_oracle_topk_optimized` package to be importable. "
            "Make sure its containing directory is on PYTHONPATH. "
            f"Original error: {e}"
        ) from e
    return SparseOracleTopKDecodeWrapper


class _SparseDecodeWrapperAdapter:
    """Thin adapter around the sparse oracle top-k wrapper.

    The sparse wrapper inherits from ``original_optimized``, not from
    ``flashinfer.BatchDecodeWithPagedKVCacheWrapper``. vLLM's
    ``fast_plan_decode`` short-circuits to FlashInfer-internal
    ``flashinfer.decode.fast_decode_plan`` when the wrapper reports
    ``is_cuda_graph_enabled=True``, and that helper probes private
    FlashInfer wrapper attributes (``_int_workspace_buffer``,
    ``_cached_module``) that the sparse wrapper does not have. This
    adapter exists solely to report ``is_cuda_graph_enabled=False`` to
    ``fast_plan_decode`` so it always falls back to ``self.plan(...)``,
    which forwards to the inner sparse wrapper's own cudagraph-aware
    ``plan()``. The inner wrapper still uses its persistent paged-KV
    buffers because it was constructed with ``use_cuda_graph=True``.

    All other attribute access (``_window_left``, ``_sm_scale``,
    ``_logits_soft_cap``, ``run``, ``plan``, ...) is forwarded
    transparently to the inner wrapper. ``top_k`` / ``channel_num`` are
    configured on the inner wrapper via its constructor, so the impl's
    ``decode_wrapper.run(q, kv_cache, k_scale=..., v_scale=..., out=...)``
    call works unchanged.
    """

    is_cuda_graph_enabled: ClassVar[bool] = False

    def __init__(self, inner: Any) -> None:
        object.__setattr__(self, "_inner", inner)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class FlashInferSparseBackend(FlashInferBackend):
    """FlashInfer backend with oracle top-k sparse paged-KV decode.

    Identical to :class:`FlashInferBackend` for prefill, cascade
    attention and the KV-cache update path; only the per-step decode
    wrapper is swapped for the sparse oracle top-k wrapper.
    """

    # The sparse decode kernel only supports unquantized fp16/bf16 KV
    # caches today (no fp8/fp4). Keep prefill capabilities matching
    # FlashInferBackend; constrain only the kv-cache types.
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # Mirror FLASHINFER (16/32/64).
        return [16, 32, 64]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # The sparse score/decode kernels are templated over
        # head_dim in {64, 128, 256}.
        return [64, 128, 256]

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # Mirror FlashInfer's supported range; the sparse decode kernel
        # uses standard CUDA features available on Turing and later.
        return capability >= DeviceCapability(7, 5) and capability <= DeviceCapability(
            12, 1
        )

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_SPARSE"

    @staticmethod
    def get_impl_cls() -> type["FlashInferSparseImpl"]:
        return FlashInferSparseImpl

    @staticmethod
    def get_builder_cls() -> type["FlashInferSparseMetadataBuilder"]:
        return FlashInferSparseMetadataBuilder


class FlashInferSparseMetadataBuilder(FlashInferMetadataBuilder):
    """Same metadata as :class:`FlashInferMetadataBuilder`, but the
    decode wrapper is the oracle top-k sparse wrapper.

    Forces the FlashInfer-native (non-TRTLLM) decode path so the sparse
    wrapper is actually exercised. The sparse wrapper is sync-free /
    CUDA-graph capturable when ``max_seq_len`` is supplied to its
    constructor (we pass ``model_config.max_model_len``), so the parent
    builder's full-cudagraph path is reused unchanged.
    """

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        attn_cfg = vllm_config.attention_config
        if attn_cfg.topk is None or attn_cfg.topk <= 0:
            raise ValueError(
                "FLASHINFER_SPARSE backend requires `attention_config.topk` "
                "to be a positive int. Pass it via "
                "`--attention-config.topk=<int>` on the CLI or "
                "`attention_config={'topk': <int>}` in Python."
            )

        self.sparse_top_k: int = int(attn_cfg.topk)
        # `-1` means "use the full head_dim". Resolve it now that we know
        # head_dim so the adapter forwards a concrete value to the kernel.
        if attn_cfg.channel_num == -1:
            self.sparse_channel_num: int = int(self.head_dim)
        else:
            self.sparse_channel_num = int(attn_cfg.channel_num)

        # Force the FI-native decode path; the sparse wrapper replaces
        # the dense decode kernel and TRTLLM has no equivalent here.
        self.use_trtllm_decode_attention = False

        # The sparse decode kernel only handles ``q_len_per_req == 1``.
        # Reset the reorder threshold so spec-decode is not bundled in.
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)

        # Engine-wide upper bound on per-request length; passed to every
        # sparse wrapper so it can use a static score-tensor last dim and
        # skip the device-side L_max computation. This is what makes the
        # decode end-to-end CUDA-graph capturable.
        self._sparse_max_seq_len: int = int(self.model_config.max_model_len)

        self._sparse_wrapper_cls: type[Any] | None = None

    def _make_sparse_decode_wrapper(
        self,
        use_cudagraph: bool,
        paged_kv_indptr: torch.Tensor | None,
        paged_kv_indices: torch.Tensor | None,
        paged_kv_last_page_len: torch.Tensor | None,
    ) -> Any:
        if self._sparse_wrapper_cls is None:
            self._sparse_wrapper_cls = _import_sparse_oracle_topk_wrapper_cls()
        return self._sparse_wrapper_cls(
            self._get_workspace_buffer(),
            get_kv_cache_layout(),
            use_cuda_graph=use_cudagraph,
            paged_kv_indptr_buffer=paged_kv_indptr,
            paged_kv_indices_buffer=paged_kv_indices,
            paged_kv_last_page_len_buffer=paged_kv_last_page_len,
            use_tensor_cores=True,
            top_k=self.sparse_top_k,
            channel_num=self.sparse_channel_num,
            max_seq_len=self._sparse_max_seq_len,
        )

    @override  # type: ignore[misc]
    def _get_decode_wrapper(
        self, batch_size: int, use_cudagraph: bool = False
    ) -> Any:
        # Mirror ``FlashInferMetadataBuilder._get_decode_wrapper`` but build
        # the sparse wrapper instead of the dense one. When ``use_cudagraph``
        # is requested, slice the persistent paged-KV buffers and cache one
        # wrapper per captured batch size, exactly like the parent.
        if use_cudagraph:
            decode_wrapper = self._decode_wrappers_cudagraph.get(batch_size, None)
        else:
            decode_wrapper = self._decode_wrapper

        if decode_wrapper is None:
            if use_cudagraph:
                paged_kv_indptr = self.paged_kv_indptr.gpu[: batch_size + 1]
                paged_kv_indices = self.paged_kv_indices.gpu
                paged_kv_last_page_len = self.paged_kv_last_page_len.gpu[:batch_size]
            else:
                paged_kv_indptr = None
                paged_kv_indices = None
                paged_kv_last_page_len = None

            inner = self._make_sparse_decode_wrapper(
                use_cudagraph=use_cudagraph,
                paged_kv_indptr=paged_kv_indptr,
                paged_kv_indices=paged_kv_indices,
                paged_kv_last_page_len=paged_kv_last_page_len,
            )
            decode_wrapper = _SparseDecodeWrapperAdapter(inner)

            if use_cudagraph:
                self._decode_wrappers_cudagraph[batch_size] = decode_wrapper  # type: ignore[assignment]
            else:
                self._decode_wrapper = decode_wrapper  # type: ignore[assignment]

        return decode_wrapper

    @override  # type: ignore[misc]
    @classmethod
    def get_cudagraph_support(
        cls: type["FlashInferSparseMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        # We always force the FI-native (non-TRTLLM) decode path, and the
        # sparse wrapper only handles q_len_per_req == 1, so the right
        # support level is single-token decode (matches FlashInfer's
        # non-TRTLLM branch).
        return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE


class FlashInferSparseImpl(FlashInferImpl):
    """Identical to :class:`FlashInferImpl`. The sparse swap happens at
    the wrapper level inside :class:`FlashInferSparseMetadataBuilder`,
    so the dispatch in ``FlashInferImpl.forward()`` is reused unchanged.
    """
