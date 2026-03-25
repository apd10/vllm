# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Literal

from pydantic import field_validator, model_validator

from vllm.config.utils import config, get_hash_factors, hash_factors

SparseAttentionType = Literal["none", "pqcache", "pqcache_sample"]

_PQ_FIELDS = (
    "pq_group_factor",
    "pq_bits",
    "pq_kmeans_iter",
    "pq_init_offset",
    "pq_metric",
)

_PQS_FIELDS = (
    "pqs_group_factor",
    "pqs_bits",
    "pqs_kmeans_iter",
    "pqs_init_offset",
    "pqs_metric",
    "pqs_base_sample_size",
)


@config
class SkylightConfig:
    """Configuration for Skylight sparse attention."""

    use_sparse_attention: bool = False
    """Master switch. When False the Skylight code path is not invoked at all.
    When True with sparse_attention_type='none', a dry-run is performed
    (full attention through the Skylight code path)."""

    sparse_attention_type: SparseAttentionType = "none"
    """Sparse attention algorithm to use."""

    # --- PQCache options (required when sparse_attention_type == "pqcache") ---
    pq_group_factor: int | None = None
    """PQCache group factor."""
    pq_bits: int | None = None
    """PQCache quantization bits."""
    pq_kmeans_iter: int | None = None
    """PQCache k-means iterations."""
    pq_init_offset: int | None = None
    """PQCache initialization offset."""
    pq_metric: str | None = None
    """PQCache distance metric."""

    # --- PQCache-Sample options (required when type == "pqcache_sample") ---
    pqs_group_factor: int | None = None
    """PQCache-Sample group factor."""
    pqs_bits: int | None = None
    """PQCache-Sample quantization bits."""
    pqs_kmeans_iter: int | None = None
    """PQCache-Sample k-means iterations."""
    pqs_init_offset: int | None = None
    """PQCache-Sample initialization offset."""
    pqs_metric: str | None = None
    """PQCache-Sample distance metric."""
    pqs_base_sample_size: float | int | None = None
    """PQCache-Sample base sample size. Float in (0, 1] means fraction of
    tokens; int >= 1 means absolute token count."""

    @field_validator("pqs_base_sample_size")
    @classmethod
    def _validate_sample_size(cls, value: Any) -> Any:
        if value is None:
            return value
        if isinstance(value, float):
            if not (0.0 < value <= 1.0):
                raise ValueError(
                    f"pqs_base_sample_size as float must be in (0.0, 1.0], "
                    f"got {value}"
                )
        elif isinstance(value, int):
            if value < 1:
                raise ValueError(
                    f"pqs_base_sample_size as int must be >= 1, got {value}"
                )
        return value

    @model_validator(mode="after")
    def _validate_field_groups(self) -> "SkylightConfig":
        pq_values = {f: getattr(self, f) for f in _PQ_FIELDS}
        pqs_values = {f: getattr(self, f) for f in _PQS_FIELDS}
        any_pq_set = any(v is not None for v in pq_values.values())
        any_pqs_set = any(v is not None for v in pqs_values.values())
        all_pq_set = all(v is not None for v in pq_values.values())
        all_pqs_set = all(v is not None for v in pqs_values.values())

        if not self.use_sparse_attention:
            if self.sparse_attention_type != "none":
                raise ValueError(
                    "Cannot select a sparse attention type when "
                    "use_sparse_attention is False"
                )
            return self

        if self.sparse_attention_type == "none":
            if any_pq_set or any_pqs_set:
                raise ValueError(
                    "No pq_*/pqs_* fields should be set in dry-run mode "
                    "(sparse_attention_type='none')"
                )
            return self

        if self.sparse_attention_type == "pqcache":
            if not all_pq_set:
                missing = [f for f, v in pq_values.items() if v is None]
                raise ValueError(
                    f"All pq_* fields are required when "
                    f"sparse_attention_type is 'pqcache'. "
                    f"Missing: {missing}"
                )
            if any_pqs_set:
                raise ValueError(
                    "pqs_* fields should only be set when "
                    "sparse_attention_type is 'pqcache_sample'"
                )

        if self.sparse_attention_type == "pqcache_sample":
            if not all_pqs_set:
                missing = [f for f, v in pqs_values.items() if v is None]
                raise ValueError(
                    f"All pqs_* fields are required when "
                    f"sparse_attention_type is 'pqcache_sample'. "
                    f"Missing: {missing}"
                )
            if any_pq_set:
                raise ValueError(
                    "pq_* fields should only be set when "
                    "sparse_attention_type is 'pqcache'"
                )

        return self

    def compute_hash(self) -> str:
        ignored_factors: set[str] = set()
        factors = get_hash_factors(self, ignored_factors)
        return hash_factors(factors)
