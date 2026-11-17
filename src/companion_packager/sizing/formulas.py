"""Mathematical hardware sizing, memory budget, and throughput formulas."""

from typing import Dict, Optional
from companion_packager.schemas import (
    CandidateModel,
    ConcurrencyScenario,
    ConcurrencyScenarioBudget,
    HardwareProfile,
    ProcessMemorySummary,
    SizingEvaluation,
)
from companion_packager.hardware.process_inspector import audit_active_processes

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


def calculate_budget_ceiling_mb(
    hardware: HardwareProfile, clean_state: bool = True
) -> float:
    """Calculate host memory budget ceiling in Megabytes.

    GPU offloading:
        min(VRAM_total * 0.80, VRAM_free * 0.95) when dynamic free VRAM is available,
        falling back to VRAM_total * 0.80.

    CPU execution:
        Enforces Agent Companion Clean-State invariant (RAM_CEILING_MB = 6144.0):
        If clean_state is True: min(RAM_total * 0.70, 6144.0 MB)
        If clean_state is False: min(RAM_available * 0.80, 6144.0 MB)
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

    if clean_state:
        return min(hardware.ram_total_mb * 0.70, RAM_CEILING_MB)
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


def calculate_ram_penalty(
    runtime_mb: float,
    budget_ceiling_mb: float,
    is_ternary: bool = False,
) -> float:
    """Calculate harmonized RAM penalty across active headroom range.

    Penalty formula:
        0.0 if runtime <= 0.875 * ceiling or (is_ternary and runtime <= 5750.0)
        (runtime - 0.875 * ceiling) / (0.125 * ceiling) if 0.875 * ceiling < runtime <= ceiling
        1000.0 (disqualified) if runtime > ceiling
    """
    if budget_ceiling_mb <= 0:
        return 1000.0
    if runtime_mb > budget_ceiling_mb:
        return 1000.0
    if is_ternary and runtime_mb <= 5750.0:
        return 0.0
    if runtime_mb <= 0.875 * budget_ceiling_mb:
        return 0.0
    denom = 0.125 * budget_ceiling_mb
    return (runtime_mb - 0.875 * budget_ceiling_mb) / denom if denom > 0 else 0.0


def evaluate_sizing(
    candidate: CandidateModel,
    hardware: HardwareProfile,
    context_tokens: int = 16384,
    bits_per_weight: float = 4.5,
    sparsity_speedup: float = 0.0,
    clean_state: bool = True,
    budget_ceiling_override_mb: Optional[float] = None,
) -> SizingEvaluation:
    """Evaluate candidate model hardware feasibility, memory budget, and projected throughput."""
    dedicated_gpus = [
        g for g in hardware.gpus
        if g.vendor.lower() in ("nvidia", "amd", "apple") and g.vram_total_mb > 0
    ]
    is_gpu = bool(dedicated_gpus) and hardware.backend_target.lower() != "cpu_forced"
    target_device = "cpu" if not is_gpu else hardware.backend_target.lower()

    # Detect ternary quantization
    is_ternary = bool(
        (candidate.quant_method and candidate.quant_method.upper() in ("TERNARY", "BITNET", "COLIBRI"))
        or ("ternaryquench" in candidate.repo_id.lower())
    )

    if is_ternary:
        # Override weights to Colibri 1.49 bits rate
        weights_mb = calculate_weights_mb(candidate.param_count_billions, 1.49)
        # Set runtime overhead matching ColibriQuantizer
        overhead_mb = 400.0
        # SpikeBrain AVX-512 boost on CPU
        if hardware.has_avx512 and "cpu" in target_device.lower() and sparsity_speedup == 0.0:
            sparsity_speedup = 0.60
        # Active weight ratio remains 1.0 (dense ternary)
        active_weight_ratio = 1.0
    else:
        weights_mb = candidate.size_mb if candidate.size_mb > 0 else calculate_weights_mb(
            candidate.param_count_billions, bits_per_weight
        )
        overhead_mb = CUDA_OVERHEAD_MB if is_gpu else CPU_OVERHEAD_MB
        active_weight_ratio = candidate.active_weight_ratio

    if budget_ceiling_override_mb is not None:
        budget_ceiling_mb = budget_ceiling_override_mb
    else:
        budget_ceiling_mb = calculate_budget_ceiling_mb(hardware, clean_state=clean_state)

    # Context Clamping on CPU under 6144 MB ceiling
    if "cpu" in target_device.lower():
        if candidate.param_count_billions and candidate.param_count_billions >= 20.0:
            effective_context = min(context_tokens, 4096)
        elif candidate.param_count_billions and candidate.param_count_billions >= 6.5:
            effective_context = min(context_tokens, 8192)
        else:
            effective_context = context_tokens
    else:
        effective_context = context_tokens

    is_mla = candidate.arch_type.lower() in ("mla", "deepseek_v2", "deepseek")
    # Initial FP16 KV cache evaluation (2.0 bytes per element)
    kv_cache_mb = calculate_kv_cache_mb(
        layers=candidate.num_layers,
        kv_heads=candidate.num_kv_heads,
        head_dim=candidate.head_dim,
        context_tokens=effective_context,
        precision_bytes=2.0,
        is_mla=is_mla,
        kv_lora_rank=candidate.mla_kv_lora_rank,
        rope_head_dim=candidate.mla_rope_head_dim,
    )
    total_ram_mb = weights_mb + kv_cache_mb + overhead_mb

    # Q8_0 KV Cache Fallback Trigger:
    # If initial FP16 evaluation exceeds 0.80 * budget_ceiling_mb, compute Q8_0 fallback (1.0 byte)
    if total_ram_mb > 0.80 * budget_ceiling_mb:
        kv_q8_mb = calculate_kv_cache_mb(
            layers=candidate.num_layers,
            kv_heads=candidate.num_kv_heads,
            head_dim=candidate.head_dim,
            context_tokens=effective_context,
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
    ram_penalty = calculate_ram_penalty(total_ram_mb, budget_ceiling_mb, is_ternary=is_ternary)
    tokens_per_second = estimate_tokens_per_second(
        bandwidth_gb_s=hardware.bandwidth_gb_s,
        weights_mb=weights_mb,
        active_weight_ratio=active_weight_ratio,
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


def calculate_scenario_budgets(
    hardware: HardwareProfile,
    process_summary: Optional[ProcessMemorySummary] = None,
) -> Dict[ConcurrencyScenario, ConcurrencyScenarioBudget]:
    """Calculate operational concurrency memory budgets across Parallel, Minimal, and Dedicated modes."""
    if process_summary is None:
        process_summary = audit_active_processes()

    is_cpu = "cpu" in hardware.backend_target.lower() or not bool(hardware.gpus)

    if is_cpu:
        ceiling_host = min(hardware.ram_total_mb * 0.70, RAM_CEILING_MB)
        ram_avail = hardware.ram_avail_mb
        non_reclaimable = (
            process_summary.ides_mb + process_summary.dev_runtimes_mb + process_summary.system_mb
        )

        # Parallel Mode
        reclaimable_par = 0.0
        budget_par = round(min(ram_avail * 0.75, ceiling_host), 1)
        budget_parallel = ConcurrencyScenarioBudget(
            scenario=ConcurrencyScenario.PARALLEL,
            title="Parallel Coexistence",
            description="Runs smoothly alongside active IDEs, open browser tabs, and dev runtimes without disruption.",
            available_ram_mb=round(ram_avail, 1),
            reclaimable_ram_mb=reclaimable_par,
            budget_ceiling_mb=budget_par,
        )

        # Minimal Interference Mode
        reclaimable_min = round(0.50 * (process_summary.browsers_mb + process_summary.other_mb), 1)
        effective_avail = min(
            ram_avail + reclaimable_min,
            max(1024.0, hardware.ram_total_mb - non_reclaimable),
        )
        budget_min = round(min(effective_avail * 0.80, ceiling_host), 1)
        budget_minimal = ConcurrencyScenarioBudget(
            scenario=ConcurrencyScenario.MINIMAL_INTERFERENCE,
            title="Minimal Interference",
            description="Reclaims ~50% dormant browser and background memory while keeping developer IDEs intact.",
            available_ram_mb=round(ram_avail, 1),
            reclaimable_ram_mb=reclaimable_min,
            budget_ceiling_mb=budget_min,
        )

        # Dedicated Alone Mode
        reclaimable_ded = round(process_summary.browsers_mb + process_summary.other_mb, 1)
        max_unencumbered = max(ram_avail, hardware.ram_total_mb - non_reclaimable) * 0.85
        budget_ded = round(min(ceiling_host, max_unencumbered), 1)
        budget_dedicated = ConcurrencyScenarioBudget(
            scenario=ConcurrencyScenario.DEDICATED,
            title="Dedicated Alone",
            description="Maximum capacity assuming background apps are closed and workstation is dedicated to Assistant.",
            available_ram_mb=round(ram_avail, 1),
            reclaimable_ram_mb=reclaimable_ded,
            budget_ceiling_mb=budget_ded,
        )

    else:
        primary_gpu = max(hardware.gpus, key=lambda g: g.vram_total_mb)
        has_free_vram = primary_gpu.vram_free_mb is not None and primary_gpu.vram_free_mb > 0
        vram_free = (
            primary_gpu.vram_free_mb
            if has_free_vram
            else primary_gpu.vram_total_mb * 0.80
        )
        ceiling_host = (
            min(primary_gpu.vram_total_mb * 0.80, vram_free * 0.95)
            if has_free_vram
            else primary_gpu.vram_total_mb * 0.80
        )

        # Parallel
        reclaimable_par = 0.0
        budget_par = round(min(vram_free * 0.80, ceiling_host), 1)
        budget_parallel = ConcurrencyScenarioBudget(
            scenario=ConcurrencyScenario.PARALLEL,
            title="Parallel Coexistence",
            description="Runs smoothly alongside active desktop applications within free VRAM allocation.",
            available_ram_mb=round(vram_free, 1),
            reclaimable_ram_mb=reclaimable_par,
            budget_ceiling_mb=budget_par,
        )

        # Minimal Interference
        reclaimable_min = 0.0
        budget_min = round(min(vram_free * 0.90, ceiling_host), 1)
        budget_minimal = ConcurrencyScenarioBudget(
            scenario=ConcurrencyScenario.MINIMAL_INTERFERENCE,
            title="Minimal Interference",
            description="Optimized VRAM allocation assuming minor GPU background processes can yield memory.",
            available_ram_mb=round(vram_free, 1),
            reclaimable_ram_mb=reclaimable_min,
            budget_ceiling_mb=budget_min,
        )

        # Dedicated Alone
        reclaimable_ded = 0.0
        budget_ded = round(ceiling_host, 1)
        budget_dedicated = ConcurrencyScenarioBudget(
            scenario=ConcurrencyScenario.DEDICATED,
            title="Dedicated Alone",
            description="Maximum capacity assuming GPU is dedicated solely to Assistant inference.",
            available_ram_mb=round(vram_free, 1),
            reclaimable_ram_mb=reclaimable_ded,
            budget_ceiling_mb=budget_ded,
        )

    return {
        ConcurrencyScenario.PARALLEL: budget_parallel,
        ConcurrencyScenario.MINIMAL_INTERFERENCE: budget_minimal,
        ConcurrencyScenario.DEDICATED: budget_dedicated,
    }
