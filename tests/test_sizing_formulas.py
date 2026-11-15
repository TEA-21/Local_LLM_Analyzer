"""Unit and integration tests for mathematical sizing formulas and Phase 9 Colibri/SpikeBrain harmonization."""

import pytest
from companion_packager.schemas import CandidateModel, HardwareProfile
from companion_packager.sizing.formulas import (
    calculate_budget_ceiling_mb,
    calculate_kv_cache_mb,
    calculate_ram_penalty,
    calculate_weights_mb,
    estimate_tokens_per_second,
    evaluate_sizing,
)
from tests.test_formulas import *  # noqa: F401, F403


def test_colibri_ternary_27b_weights_and_overhead():
    """Verify Colibri 1.49 bits/param calculation and 400MB runtime overhead.

    Weights for 27B at 1.49 bits:
        (27.0 * 1e9 * 1.49) / (8 * 1024^2) = 4795.833587... MB <= 4800.0 MB.
    """
    raw_weights = calculate_weights_mb(27.0, 1.49)
    assert raw_weights <= 4800.0
    assert pytest.approx(raw_weights, rel=1e-4) == 4795.8336

    cand = CandidateModel(
        repo_id="penkia/TernaryQuench-Qwen3.8-27B-GGUF",
        filename="TernaryQuench-Qwen3.8-27B-Q2_K.gguf",
        quant_method="TERNARY",
        param_count_billions=27.0,
        size_mb=4796.0,
        context_window=32768,
        num_layers=64,
        num_kv_heads=8,
        head_dim=128,
    )

    hw = HardwareProfile(
        os_name="windows",
        cpu_brand="Intel Core i7-1165G7",
        cpu_arch="x86_64",
        physical_cores=4,
        logical_cores=8,
        has_avx512=True,
        has_avx2=True,
        has_neon=False,
        ram_total_mb=16384.0,
        ram_avail_mb=12000.0,
        disk_free_gb=100.0,
        gpus=[],
        bandwidth_gb_s=30.0,
        backend_target="cpu",
    )

    eval_res = evaluate_sizing(cand, hw, context_tokens=16384)
    assert pytest.approx(eval_res.weights_mb, abs=0.1) == 4795.83
    assert eval_res.weights_mb == round(raw_weights, 2)
    assert eval_res.weights_mb <= 4800.0
    assert eval_res.overhead_mb == 400.0
    # Combined model envelope must fit under 5200MB normal budget
    assert eval_res.weights_mb + eval_res.overhead_mb <= 5200.0


def test_clean_state_cpu_budget_ceiling_16gb_and_8gb():
    """Verify clean-state CPU budget ceiling enforcement:
    16GB host -> min(16384 * 0.70, 6144.0) = 6144.0 MB.
    8GB host -> min(8192 * 0.70, 6144.0) = 5734.4 MB.
    Fallback with clean_state=False -> min(avail * 0.80, 6144.0).
    """
    hw_16gb = HardwareProfile(
        os_name="windows",
        cpu_brand="Intel Core i7-1165G7",
        cpu_arch="x86_64",
        physical_cores=4,
        logical_cores=8,
        has_avx512=True,
        has_avx2=True,
        has_neon=False,
        ram_total_mb=16384.0,
        ram_avail_mb=10000.0,
        disk_free_gb=100.0,
        gpus=[],
        bandwidth_gb_s=30.0,
        backend_target="cpu",
    )
    # Default clean_state is True
    assert calculate_budget_ceiling_mb(hw_16gb) == 6144.0
    assert calculate_budget_ceiling_mb(hw_16gb, clean_state=True) == 6144.0

    hw_8gb = HardwareProfile(
        os_name="windows",
        cpu_brand="Intel Core i5-10210U",
        cpu_arch="x86_64",
        physical_cores=4,
        logical_cores=8,
        has_avx512=False,
        has_avx2=True,
        has_neon=False,
        ram_total_mb=8192.0,
        ram_avail_mb=5000.0,
        disk_free_gb=100.0,
        gpus=[],
        bandwidth_gb_s=25.0,
        backend_target="cpu",
    )
    assert calculate_budget_ceiling_mb(hw_8gb) == 5734.4
    assert calculate_budget_ceiling_mb(hw_8gb, clean_state=True) == 5734.4

    # With clean_state=False, falls back to min(avail * 0.80, 6144.0)
    assert calculate_budget_ceiling_mb(hw_8gb, clean_state=False) == 4000.0


def test_spikebrain_avx512_bandwidth_acceleration():
    """Verify SpikeBrain AVX-512 neuromorphic sparsity boost:
    10.0 tok/s on 30.0 GB/s DDR4 memory bus with S = 0.60.
    """
    cand = CandidateModel(
        repo_id="penkia/TernaryQuench-Qwen3.8-27B-GGUF",
        filename="TernaryQuench-Qwen3.8-27B-Q2_K.gguf",
        quant_method="TERNARY",
        param_count_billions=27.0,
        size_mb=4796.0,
        context_window=32768,
        num_layers=64,
        num_kv_heads=8,
        head_dim=128,
    )

    hw_avx512 = HardwareProfile(
        os_name="windows",
        cpu_brand="Intel Core i7-1165G7",
        cpu_arch="x86_64",
        physical_cores=4,
        logical_cores=8,
        has_avx512=True,
        has_avx2=True,
        has_neon=False,
        ram_total_mb=16384.0,
        ram_avail_mb=12000.0,
        disk_free_gb=100.0,
        gpus=[],
        bandwidth_gb_s=30.0,
        backend_target="cpu",
    )

    eval_res = evaluate_sizing(cand, hw_avx512, context_tokens=16384)
    # 30.0 * 1.6 / (4795.83 / 1024) ≈ 10.25 tok/s (within 9.5-12.0 tok/s range)
    assert 9.5 <= eval_res.tokens_per_second <= 12.0
    assert pytest.approx(eval_res.tokens_per_second, abs=0.5) == 10.0


def test_context_clamping_27b_and_7b_cpu():
    """Verify context clamping under 6144 MB CPU ceiling:
    - 27B model: context clamped to 4096 (KV cache = 512 MB Q8_0).
    - 7B model: context clamped to 8192 (KV cache <= 512 MB).
    """
    hw = HardwareProfile(
        os_name="windows",
        cpu_brand="Intel Core i7-1165G7",
        cpu_arch="x86_64",
        physical_cores=4,
        logical_cores=8,
        has_avx512=True,
        has_avx2=True,
        has_neon=False,
        ram_total_mb=16384.0,
        ram_avail_mb=12000.0,
        disk_free_gb=100.0,
        gpus=[],
        bandwidth_gb_s=30.0,
        backend_target="cpu",
    )

    # 27B Ternary Model: 64 layers, 8 kv heads, head dim 128
    cand_27b = CandidateModel(
        repo_id="penkia/TernaryQuench-Qwen3.8-27B-GGUF",
        filename="TernaryQuench-Qwen3.8-27B-Q2_K.gguf",
        quant_method="TERNARY",
        param_count_billions=27.0,
        size_mb=4796.0,
        context_window=32768,
        num_layers=64,
        num_kv_heads=8,
        head_dim=128,
    )
    eval_27b = evaluate_sizing(cand_27b, hw, context_tokens=16384)
    # Effective context clamped to 4096. Initial FP16 KV = 1024MB triggers Q8_0 -> 512MB
    assert eval_27b.kv_cache_mb == 512.0
    expected_total_27b = round(eval_27b.weights_mb + 512.0 + 400.0, 2)
    assert eval_27b.total_ram_mb == expected_total_27b
    assert pytest.approx(eval_27b.total_ram_mb, abs=0.1) == 5707.83
    assert eval_27b.fits is True
    assert eval_27b.total_ram_mb <= 6144.0

    # 7B Dense Model: 28 layers, 4 kv heads, head dim 128, size 4680 MB
    cand_7b = CandidateModel(
        repo_id="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        filename="qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        quant_method="Q4_K_M",
        param_count_billions=7.61,
        size_mb=4680.0,
        context_window=32768,
        num_layers=28,
        num_kv_heads=4,
        head_dim=128,
    )
    eval_7b = evaluate_sizing(cand_7b, hw, context_tokens=16384)
    # Effective context clamped to 8192 -> FP16 is 448MB which triggers Q8_0 (224MB) <= 512.0 MB
    assert eval_7b.kv_cache_mb <= 512.0
    assert eval_7b.kv_cache_mb == 224.0
    assert eval_7b.total_ram_mb == 4680.0 + 224.0 + 200.0
    assert eval_7b.total_ram_mb == 5104.0
    assert eval_7b.fits is True
    assert eval_7b.total_ram_mb <= 6144.0


def test_ternary_ram_penalty_exemption_under_5750mb():
    """Verify Colibri ternary models are exempt from RAM penalty when runtime <= 5750.0 MB."""
    ceiling = 6144.0

    # Non-ternary at 5707.83 MB exceeds 0.875 * 6144.0 (5376.0 MB) -> receives non-zero penalty
    normal_penalty = calculate_ram_penalty(5707.83, ceiling, is_ternary=False)
    assert normal_penalty > 0.0
    expected_penalty = (5707.83 - 5376.0) / (0.125 * 6144.0)
    assert pytest.approx(normal_penalty, rel=1e-3) == expected_penalty

    # Ternary at 5707.83 MB <= 5750.0 MB -> 0.0 penalty
    ternary_penalty = calculate_ram_penalty(5707.83, ceiling, is_ternary=True)
    assert ternary_penalty == 0.0

    # Ternary exceeding 5750.0 MB (e.g. 5800.0 MB) receives standard gradient penalty
    over_threshold_penalty = calculate_ram_penalty(5800.0, ceiling, is_ternary=True)
    assert over_threshold_penalty > 0.0

    # Over ceiling -> 1000.0 disqualification
    assert calculate_ram_penalty(6200.0, ceiling, is_ternary=True) == 1000.0

