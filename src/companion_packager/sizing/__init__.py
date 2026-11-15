"""Hardware sizing, throughput evaluation, and recommendation package."""

from companion_packager.sizing.formulas import (
    CPU_OVERHEAD_MB,
    CUDA_OVERHEAD_MB,
    RAM_CEILING_MB,
    calculate_budget_ceiling_mb,
    calculate_kv_cache_mb,
    calculate_ram_penalty,
    calculate_weights_mb,
    estimate_tokens_per_second,
    evaluate_sizing,
)
from companion_packager.sizing.recommender import (
    PERMISSIVE_LICENSES,
    display_scorecard_table,
    rank_candidates,
)

__all__ = [
    "RAM_CEILING_MB",
    "CUDA_OVERHEAD_MB",
    "CPU_OVERHEAD_MB",
    "PERMISSIVE_LICENSES",
    "calculate_weights_mb",
    "calculate_kv_cache_mb",
    "calculate_budget_ceiling_mb",
    "estimate_tokens_per_second",
    "calculate_ram_penalty",
    "evaluate_sizing",
    "rank_candidates",
    "display_scorecard_table",
]
