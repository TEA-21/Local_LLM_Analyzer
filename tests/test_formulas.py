"""Unit tests for mathematical hardware sizing and throughput formulas."""

import pytest
from companion_packager.schemas import CandidateModel, GPUInfo, HardwareProfile, SizingEvaluation
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


def test_calculate_weights_mb():
    """Verify weight formula (P * b) / (8 * 1024^2)."""
    # 7B parameters at 4.0 bits per weight
    weights_mb = calculate_weights_mb(param_count_billions=7.0, bits_per_weight=4.0)
    # 7e9 * 4 / (8 * 1024 * 1024) = 28e9 / 8388608 ≈ 3337.86 MB
    expected_mb = (7.0 * 1e9 * 4.0) / (8.0 * 1024.0 * 1024.0)
    assert pytest.approx(weights_mb, rel=1e-4) == expected_mb
    assert 3337.0 < weights_mb < 3339.0


def test_calculate_kv_cache_gqa():
    """Verify GQA KV cache formula for Qwen2.5 7B.

    Layers: 28, KV Heads: 4, Head Dim: 128, Context: 16384, Precision: FP16 (2 bytes).
    Expected: (2 * 28 * 4 * 128 * 16384 * 2) / 1024^2 = 896.0 MB.
    """
    kv_mb = calculate_kv_cache_mb(
        layers=28,
        kv_heads=4,
        head_dim=128,
        context_tokens=16384,
        precision_bytes=2.0,
        is_mla=False,
    )
    assert pytest.approx(kv_mb, rel=1e-4) == 896.0


def test_calculate_kv_cache_mla():
    """Verify DeepSeek Multi-Head Latent Attention (MLA) KV cache formula.

    Layers: 27, kv_lora_rank (d_c): 512, rope_head_dim (d_r): 64, Context: 16384, Precision: FP16 (2 bytes).
    Formula: (L * (d_c + d_r) * C * P_kv) / 1024^2
    Expected: (27 * (512 + 64) * 16384 * 2) / 1024^2 ≈ 486.0 MB.
    """
    kv_mla_mb = calculate_kv_cache_mb(
        layers=27,
        kv_heads=0,
        head_dim=0,
        context_tokens=16384,
        precision_bytes=2.0,
        is_mla=True,
        kv_lora_rank=512,
        rope_head_dim=64,
    )
    expected_mla = (27 * (512 + 64) * 16384 * 2.0) / (1024.0 * 1024.0)
    assert pytest.approx(kv_mla_mb, rel=1e-4) == expected_mla
    assert pytest.approx(kv_mla_mb, rel=1e-3) == 486.0


def test_budget_ceiling_cpu_enforces_6144mb_ceiling():
    """Verify CPU execution invariant: min(avail_ram * 0.80, 6144.0 MB)."""
    # High-end CPU with 32GB available RAM -> capped at 6144.0 MB
    high_ram_cpu = HardwareProfile(
        os_name="windows",
        cpu_brand="Intel Core i9-13900K",
        cpu_arch="x86_64",
        physical_cores=16,
        logical_cores=32,
        has_avx512=False,
        has_avx2=True,
        has_neon=False,
        ram_total_mb=65536.0,
        ram_avail_mb=32768.0,
        disk_free_gb=100.0,
        gpus=[],
        bandwidth_gb_s=50.0,
        backend_target="cpu",
    )
    ceiling_high = calculate_budget_ceiling_mb(high_ram_cpu)
    assert ceiling_high == RAM_CEILING_MB
    assert ceiling_high == 6144.0

    # Low-end 8GB CPU with 6000MB available RAM -> 6000 * 0.80 = 4800.0 MB (< 6144.0)
    low_ram_cpu = HardwareProfile(
        os_name="linux",
        cpu_brand="Intel Core i5-8250U",
        cpu_arch="x86_64",
        physical_cores=4,
        logical_cores=8,
        has_avx512=False,
        has_avx2=True,
        has_neon=False,
        ram_total_mb=8192.0,
        ram_avail_mb=6000.0,
        disk_free_gb=40.0,
        gpus=[],
        bandwidth_gb_s=25.0,
        backend_target="cpu",
    )
    ceiling_low = calculate_budget_ceiling_mb(low_ram_cpu, clean_state=False)
    assert ceiling_low == 4800.0

    # With clean_state=True (default), 8GB host uses 8192 * 0.70 = 5734.4 MB
    ceiling_low_clean = calculate_budget_ceiling_mb(low_ram_cpu, clean_state=True)
    assert ceiling_low_clean == 5734.4


def test_budget_ceiling_four_tiers():
    """Verify exact RAM budget across 4 tiers:
    1. Low-end 8GB CPU
    2. Sub-6GB AVX-512 laptop
    3. 8GB GPU
    4. 24GB RTX 4090
    """
    # Tier 1: Low-end 8GB CPU (ram_avail = 7000 MB, no GPU)
    t1_hw = HardwareProfile(
        os_name="linux",
        cpu_brand="Intel Core i5-10210U",
        cpu_arch="x86_64",
        physical_cores=4,
        logical_cores=8,
        has_avx512=False,
        has_avx2=True,
        has_neon=False,
        ram_total_mb=8192.0,
        ram_avail_mb=7000.0,
        disk_free_gb=50.0,
        gpus=[],
    )
    # Clean-state False: min(7000 * 0.80 = 5600.0, 6144.0) = 5600.0 MB
    assert calculate_budget_ceiling_mb(t1_hw, clean_state=False) == 5600.0
    # Clean-state True (default): min(8192 * 0.70 = 5734.4, 6144.0) = 5734.4 MB
    assert calculate_budget_ceiling_mb(t1_hw, clean_state=True) == 5734.4
    assert calculate_budget_ceiling_mb(t1_hw) == 5734.4

    # Tier 2: Sub-6GB AVX-512 laptop (ram_avail = 5000 MB, AVX-512, no GPU)
    t2_hw = HardwareProfile(
        os_name="windows",
        cpu_brand="Intel Core i7-1165G7",
        cpu_arch="x86_64",
        physical_cores=4,
        logical_cores=8,
        has_avx512=True,
        has_avx2=True,
        has_neon=False,
        ram_total_mb=8192.0,
        ram_avail_mb=5000.0,
        disk_free_gb=50.0,
        gpus=[],
    )
    # Clean-state False: min(5000 * 0.80 = 4000.0, 6144.0) = 4000.0 MB
    assert calculate_budget_ceiling_mb(t2_hw, clean_state=False) == 4000.0
    # Clean-state True: min(8192 * 0.70 = 5734.4, 6144.0) = 5734.4 MB
    assert calculate_budget_ceiling_mb(t2_hw, clean_state=True) == 5734.4

    # Tier 3: 8GB GPU (RTX 3070 Mobile 8192 MB VRAM, free VRAM = 0 fallback)
    t3_hw = HardwareProfile(
        os_name="windows",
        cpu_brand="AMD Ryzen 7 5800H",
        cpu_arch="x86_64",
        physical_cores=8,
        logical_cores=16,
        has_avx512=False,
        has_avx2=True,
        has_neon=False,
        ram_total_mb=16384.0,
        ram_avail_mb=12000.0,
        disk_free_gb=100.0,
        gpus=[
            GPUInfo(
                index=0,
                name="NVIDIA GeForce RTX 3070 Laptop GPU",
                vendor="nvidia",
                vram_total_mb=8192.0,
                vram_free_mb=0.0,
            )
        ],
        backend_target="cuda",
    )
    # 8192 * 0.80 = 6553.6 MB
    assert pytest.approx(calculate_budget_ceiling_mb(t3_hw), rel=1e-4) == 6553.6

    # Tier 4: 24GB RTX 4090 (24576 MB VRAM, free VRAM = 22000 MB)
    t4_hw = HardwareProfile(
        os_name="windows",
        cpu_brand="AMD Ryzen 9 7950X",
        cpu_arch="x86_64",
        physical_cores=16,
        logical_cores=32,
        has_avx512=True,
        has_avx2=True,
        has_neon=False,
        ram_total_mb=65536.0,
        ram_avail_mb=50000.0,
        disk_free_gb=500.0,
        gpus=[
            GPUInfo(
                index=0,
                name="NVIDIA GeForce RTX 4090",
                vendor="nvidia",
                vram_total_mb=24576.0,
                vram_free_mb=22000.0,
            )
        ],
        backend_target="cuda",
    )
    # min(24576 * 0.80 = 19660.8, 22000 * 0.95 = 20900.0) = 19660.8 MB
    assert pytest.approx(calculate_budget_ceiling_mb(t4_hw), rel=1e-4) == 19660.8


def test_budget_ceiling_dynamic_vram_free_constrains():
    """Verify dynamic free VRAM safety constraint min(vram_total * 0.80, vram_free * 0.95)."""
    # 8GB GPU where only 4000 MB VRAM is free
    hw = HardwareProfile(
        os_name="windows",
        cpu_brand="AMD Ryzen 5",
        cpu_arch="x86_64",
        physical_cores=6,
        logical_cores=12,
        has_avx512=False,
        has_avx2=True,
        has_neon=False,
        ram_total_mb=16384.0,
        ram_avail_mb=10000.0,
        disk_free_gb=100.0,
        gpus=[
            GPUInfo(
                index=0,
                name="NVIDIA RTX 4060",
                vendor="nvidia",
                vram_total_mb=8192.0,
                vram_free_mb=4000.0,
            )
        ],
    )
    # min(8192 * 0.80 = 6553.6, 4000 * 0.95 = 3800.0) = 3800.0 MB
    assert calculate_budget_ceiling_mb(hw) == 3800.0


def test_estimate_tokens_per_second_moe_and_sparsity():
    """Verify bandwidth token generation formula and MoE active_weight_ratio speedup."""
    # Dense model: 9200 MB weights on 300 GB/s bus -> effective_weights_gb = 9200/1024 ≈ 8.984 GB
    # Tok/s = 300 / 8.984 ≈ 33.39 tok/s
    dense_tok_s = estimate_tokens_per_second(
        bandwidth_gb_s=300.0,
        weights_mb=9200.0,
        active_weight_ratio=1.0,
        sparsity_speedup=0.0,
    )
    assert pytest.approx(dense_tok_s, rel=1e-2) == 33.39

    # MoE model (e.g. DeepSeek-V2-Lite): 9600 MB weights, active_weight_ratio = 0.15
    # effective_weights_gb = (9600/1024) * 0.15 ≈ 1.40625 GB
    # Tok/s = 300 / 1.40625 ≈ 213.33 tok/s (MoE speedup)
    moe_tok_s = estimate_tokens_per_second(
        bandwidth_gb_s=300.0,
        weights_mb=9600.0,
        active_weight_ratio=0.15,
        sparsity_speedup=0.0,
    )
    assert pytest.approx(moe_tok_s, rel=1e-2) == 213.33

    # Sparsity speedup (SpikeBrain 0.60): 300 * (1 + 0.60) / 8.984 ≈ 53.43 tok/s
    sparse_tok_s = estimate_tokens_per_second(
        bandwidth_gb_s=300.0,
        weights_mb=9200.0,
        active_weight_ratio=1.0,
        sparsity_speedup=0.60,
    )
    assert pytest.approx(sparse_tok_s, rel=1e-2) == dense_tok_s * 1.60


def test_calculate_ram_penalty():
    """Verify harmonized RAM penalty step and gradient function."""
    ceiling = 6000.0
    # Safe range: <= 87.5% of ceiling (5250.0 MB) -> 0.0
    assert calculate_ram_penalty(5000.0, ceiling) == 0.0
    assert calculate_ram_penalty(5250.0, ceiling) == 0.0

    # Active gradient range: 5250.0 to 6000.0 MB
    # Midpoint: (5625.0 - 5250.0) / 750.0 = 0.5
    assert pytest.approx(calculate_ram_penalty(5625.0, ceiling), rel=1e-3) == 0.5

    # At exact ceiling (6000.0 MB) -> 1.0
    assert pytest.approx(calculate_ram_penalty(6000.0, ceiling), rel=1e-3) == 1.0

    # Over ceiling -> 1000.0 (disqualified)
    assert calculate_ram_penalty(6001.0, ceiling) == 1000.0


def test_evaluate_sizing_fits_and_rejects():
    """Verify evaluate_sizing end-to-end fit decision and rejection reason."""
    # Sub-6GB CPU host with 5000 MB available RAM -> budget ceiling = 4000.0 MB
    laptop_hw = HardwareProfile(
        os_name="windows",
        cpu_brand="Intel Core i7-1165G7",
        cpu_arch="x86_64",
        physical_cores=4,
        logical_cores=8,
        has_avx512=True,
        has_avx2=True,
        has_neon=False,
        ram_total_mb=8192.0,
        ram_avail_mb=5000.0,
        disk_free_gb=50.0,
        gpus=[],
        bandwidth_gb_s=30.0,
        backend_target="cpu",
    )

    # Candidate 1: Small model (3B Q4_K_M, size_mb = 2050.0, layers=28, kv_heads=4, head_dim=128)
    # Weights = 2050.0 MB, KV (16k) = 896.0 MB, CPU overhead = 200.0 MB
    # Total = 3146.0 MB <= 4000.0 MB -> FITS
    cand_small = CandidateModel(
        repo_id="Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
        filename="qwen2.5-coder-3b-instruct-q4_k_m.gguf",
        quant_method="Q4_K_M",
        param_count_billions=3.0,
        size_mb=2050.0,
        context_window=32768,
        num_layers=28,
        num_kv_heads=4,
        head_dim=128,
    )
    eval_small = evaluate_sizing(cand_small, laptop_hw, context_tokens=16384, clean_state=False)
    assert eval_small.fits is True
    assert eval_small.weights_mb == 2050.0
    assert eval_small.overhead_mb == CPU_OVERHEAD_MB
    assert eval_small.budget_ceiling_mb == 4000.0
    assert eval_small.total_ram_mb == 2050.0 + 896.0 + 200.0
    assert eval_small.rejection_reason == ""

    # Candidate 2: Big model (14B Q4_K_M, size_mb = 9200.0) -> REJECTED
    cand_large = CandidateModel(
        repo_id="Qwen/Qwen2.5-Coder-14B-Instruct-GGUF",
        filename="qwen2.5-coder-14b-instruct-q4_k_m.gguf",
        quant_method="Q4_K_M",
        param_count_billions=14.0,
        size_mb=9200.0,
        context_window=32768,
        num_layers=48,
        num_kv_heads=8,
        head_dim=128,
    )
    eval_large = evaluate_sizing(cand_large, laptop_hw, context_tokens=16384, clean_state=False)
    assert eval_large.fits is False
    assert eval_large.total_ram_mb > 4000.0
    assert "exceeds budget ceiling" in eval_large.rejection_reason
    assert eval_large.ram_penalty == 1000.0


def test_evaluate_sizing_kv_cache_q8_fallback():
    """Verify KV cache automatically falls back to Q8_0 (1.0 byte) if FP16 (2.0 bytes) marginally exceeds ceiling."""
    # Host budget ceiling: exactly 3500.0 MB
    hw = HardwareProfile(
        os_name="linux",
        cpu_brand="Intel",
        cpu_arch="x86_64",
        physical_cores=4,
        logical_cores=8,
        has_avx512=False,
        has_avx2=True,
        has_neon=False,
        ram_total_mb=8192.0,
        ram_avail_mb=4375.0,  # 4375 * 0.80 = 3500.0 MB ceiling
        disk_free_gb=50.0,
        gpus=[],
        backend_target="cpu",
    )
    # Weights = 2500.0 MB, overhead = 200.0 MB -> budget left for KV = 800.0 MB
    # With FP16: KV = 896.0 MB -> total = 3596.0 MB > 3500.0 MB (fails FP16)
    # With Q8_0: KV = 448.0 MB -> total = 3148.0 MB <= 3500.0 MB (fits Q8_0)
    cand = CandidateModel(
        repo_id="test/model",
        filename="test.gguf",
        quant_method="Q4_K_M",
        param_count_billions=3.5,
        size_mb=2500.0,
        context_window=16384,
        num_layers=28,
        num_kv_heads=4,
        head_dim=128,
    )
    eval_res = evaluate_sizing(cand, hw, context_tokens=16384, clean_state=False)
    assert eval_res.fits is True
    assert eval_res.kv_cache_mb == 448.0
    assert eval_res.total_ram_mb == 2500.0 + 448.0 + 200.0
