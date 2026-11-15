"""Mathematical hardware sizing, memory budget, and throughput formulas."""

from typing import Optional
from companion_packager.schemas import CandidateModel, HardwareProfile, SizingEvaluation

RAM_CEILING_MB: float = 6144.0
CUDA_OVERHEAD_MB: float = 500.0
CPU_OVERHEAD_MB: float = 200.0


def calculate_weights_mb(param_count_billions: float, bits_per_weight: float) -> float:
    """Calculate model weight footprint in Megabytes.

    Formula:
        M_weights = (P * b) / (8 * 1024^2)
    where P is total parameter count and b is bits per weight.
    """
    return (param_count_billions * 1e9 * bits_per_weight) / (8.0 * 1024.0 * 1024.0)


def calculate_kv_cache_mb(
    layers: int,
    kv_heads: int,
    head_dim: int,
    context_tokens: int,
    precision_bytes: float = 2.0,
    is_mla: bool = False,
    kv_lora_rank: int = 512,
    rope_head_dim: int = 64,
) -> float:
    """Calculate Key-Value cache memory requirement in Megabytes.

    For standard Multi-Head / Grouped-Query Attention (MHA/GQA):
        M_kv = (2 * L * H_kv * D_head * C * P_kv) / (1024^2)

    For Multi-Head Latent Attention (MLA, e.g. DeepSeek-Coder-V2):
        M_kv = (L * (d_c + d_r) * C * P_kv) / (1024^2)
    where d_c is kv_lora_rank and d_r is rope_head_dim.
    """
    if is_mla:
        return (layers * (kv_lora_rank + rope_head_dim) * context_tokens * precision_bytes) / (1024.0 * 1024.0)
    return (2.0 * layers * kv_heads * head_dim * context_tokens * precision_bytes) / (1024.0 * 1024.0)


def calculate_budget_ceiling_mb(hardware: HardwareProfile) -> float:
    """Calculate host memory budget ceiling in Megabytes.

    GPU offloading:
        min(VRAM_total * 0.80, VRAM_free * 0.95) when dynamic free VRAM is available,
        falling back to VRAM_total * 0.80.

    CPU execution:
        Enforces Agent Companion invariant (RAM_CEILING_MB = 6144.0):
        min(RAM_available * 0.80, 6144.0 MB).
    """
    dedicated_gpus = [
        g for g in hardware.gpus
        if g.vendor.lower() in ("nvidia", "amd", "apple") and g.vram_total_mb > 0
    ]
    if dedicated_gpus and hardware.backend_target.lower() != "cpu_forced":
        primary_gpu = max(dedicated_gpus, key=lambda g: g.vram_total_mb)
        if primary_gpu.vram_free_mb > 0:
            return min(primary_gpu.vram_total_mb * 0.80, primary_gpu.vram_free_mb * 0.95)
        return primary_gpu.vram_total_mb * 0.80

    return min(hardware.ram_avail_mb * 0.80, RAM_CEILING_MB)


def estimate_tokens_per_second(
    bandwidth_gb_s: float,
    weights_mb: float,
    active_weight_ratio: float = 1.0,
    sparsity_speedup: float = 0.0,
    sparsity_factor: Optional[float] = None,
) -> float:
    """Estimate autoregressive token generation throughput bound by memory bandwidth.

    Formula:
        Tokens/sec = (B_sustained * (1 + S)) / max((M_weights_GB * effective_ratio), 0.001)
    """
    sparsity = sparsity_factor if sparsity_factor is not None else sparsity_speedup
    clamped_sparsity = max(0.0, min(float(sparsity), 1.0))
    effective_ratio = max(active_weight_ratio, 0.05)
    weights_gb = weights_mb / 1024.0
    denominator = max(weights_gb * effective_ratio, 0.001)
    return (bandwidth_gb_s * (1.0 + clamped_sparsity)) / denominator


def calculate_ram_penalty(runtime_mb: float, budget_ceiling_mb: float) -> float:
    """Calculate harmonized RAM penalty across active headroom range.

    Penalty formula:
        0.0 if runtime <= 0.875 * ceiling
        (runtime - 0.875 * ceiling) / (0.125 * ceiling) if 0.875 * ceiling < runtime <= ceiling
        1000.0 (disqualified) if runtime > ceiling
    """
    if budget_ceiling_mb <= 0:
        return 1000.0
    if runtime_mb <= 0.875 * budget_ceiling_mb:
        return 0.0
    if runtime_mb <= budget_ceiling_mb:
        denom = 0.125 * budget_ceiling_mb
        return (runtime_mb - 0.875 * budget_ceiling_mb) / denom if denom > 0 else 0.0
    return 1000.0


def evaluate_sizing(
    candidate: CandidateModel,
    hardware: HardwareProfile,
    context_tokens: int,
    bits_per_weight: float = 4.5,
    sparsity_speedup: float = 0.0,
) -> SizingEvaluation:
    """Evaluate candidate model hardware feasibility, memory budget, and projected throughput."""
    weights_mb = candidate.size_mb if candidate.size_mb > 0 else calculate_weights_mb(
        candidate.param_count_billions, bits_per_weight
    )

    dedicated_gpus = [
        g for g in hardware.gpus
        if g.vendor.lower() in ("nvidia", "amd", "apple") and g.vram_total_mb > 0
    ]
    is_gpu = bool(dedicated_gpus) and hardware.backend_target.lower() != "cpu_forced"
    overhead_mb = CUDA_OVERHEAD_MB if is_gpu else CPU_OVERHEAD_MB
    budget_ceiling_mb = calculate_budget_ceiling_mb(hardware)

    is_mla = candidate.arch_type.lower() in ("mla", "deepseek_v2", "deepseek")
    # Attempt FP16 KV cache (2.0 bytes per element)
    kv_cache_mb = calculate_kv_cache_mb(
        layers=candidate.num_layers,
        kv_heads=candidate.num_kv_heads,
        head_dim=candidate.head_dim,
        context_tokens=context_tokens,
        precision_bytes=2.0,
        is_mla=is_mla,
        kv_lora_rank=candidate.mla_kv_lora_rank,
        rope_head_dim=candidate.mla_rope_head_dim,
    )
    total_ram_mb = weights_mb + kv_cache_mb + overhead_mb

    # If exceeding budget with FP16, fall back to Q8_0 KV cache (1.0 byte per element)
    if total_ram_mb > budget_ceiling_mb:
        kv_q8_mb = calculate_kv_cache_mb(
            layers=candidate.num_layers,
            kv_heads=candidate.num_kv_heads,
            head_dim=candidate.head_dim,
            context_tokens=context_tokens,
            precision_bytes=1.0,
            is_mla=is_mla,
            kv_lora_rank=candidate.mla_kv_lora_rank,
            rope_head_dim=candidate.mla_rope_head_dim,
        )
        total_q8_mb = weights_mb + kv_q8_mb + overhead_mb
        if total_q8_mb <= budget_ceiling_mb:
            kv_cache_mb = kv_q8_mb
            total_ram_mb = total_q8_mb

    fits = total_ram_mb <= budget_ceiling_mb
    rejection_reason = (
        ""
        if fits
        else f"Runtime memory ({total_ram_mb:.1f} MB) exceeds budget ceiling ({budget_ceiling_mb:.1f} MB)"
    )
    ram_penalty = calculate_ram_penalty(total_ram_mb, budget_ceiling_mb)
    tokens_per_second = estimate_tokens_per_second(
        bandwidth_gb_s=hardware.bandwidth_gb_s,
        weights_mb=weights_mb,
        active_weight_ratio=candidate.active_weight_ratio,
        sparsity_speedup=sparsity_speedup,
    )

    return SizingEvaluation(
        fits=fits,
        weights_mb=round(weights_mb, 2),
        kv_cache_mb=round(kv_cache_mb, 2),
        overhead_mb=round(overhead_mb, 2),
        total_ram_mb=round(total_ram_mb, 2),
        budget_ceiling_mb=round(budget_ceiling_mb, 2),
        tokens_per_second=round(tokens_per_second, 2),
        ram_penalty=round(ram_penalty, 4),
        rejection_reason=rejection_reason,
    )
