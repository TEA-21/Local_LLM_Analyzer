"""Unit and integration tests for research subsystem, benchmarks, HF client, and citations."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from companion_packager.research.benchmarks import (
    clamp,
    load_benchmarks_cache,
    normalize_benchmarks,
)
from companion_packager.research.citations import (
    generate_citation_card,
    generate_recommendations_markdown,
)
from companion_packager.research.hf_client import get_model_candidates
from companion_packager.schemas import (
    CandidateModel,
    GPUInfo,
    HardwareProfile,
    Scorecard,
    SizingEvaluation,
    UserNeeds,
)


def test_load_benchmarks_cache_success():
    """Verify that offline benchmarks_cache.json loads valid candidate models."""
    candidates = load_benchmarks_cache()
    assert len(candidates) == 20

    repos = {c.repo_id for c in candidates}
    expected_repos = {
        "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
        "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
        "bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
        "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF",
        "penkia/TernaryQuench-Qwen3.8-27B-GGUF",
        "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF",
        "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "Qwen/Qwen2.5-14B-Instruct-GGUF",
        "unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF",
        "unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF",
    }
    assert expected_repos.issubset(repos)

    # Inspect specific model attributes
    deepseek = next(c for c in candidates if "DeepSeek-Coder-V2-Lite" in c.repo_id)
    assert deepseek.arch_type == "deepseek_v2"
    assert deepseek.active_weight_ratio == 0.15
    assert deepseek.mla_v_head_dim == 128
    assert deepseek.mla_kv_lora_rank == 512
    assert deepseek.license == "deepseek-license"

    ternary = next(c for c in candidates if "TernaryQuench" in c.repo_id)
    assert ternary.quant_method in ("TERNARY", "Q2_K")
    assert ternary.param_count_billions == 27.0

    # Verify newly added Q3_K_M multi-tier models
    q3_models = [c for c in candidates if c.quant_method == "Q3_K_M"]
    assert len(q3_models) == 5
    q3_filenames = {c.filename for c in q3_models}
    assert "Meta-Llama-3.1-8B-Instruct-Q3_K_M.gguf" in q3_filenames
    assert "qwen2.5-7b-instruct-q3_k_m.gguf" in q3_filenames
    assert "qwen2.5-14b-instruct-q3_k_m.gguf" in q3_filenames
    assert "DeepSeek-R1-Distill-Qwen-7B-Q3_K_M.gguf" in q3_filenames
    assert "DeepSeek-R1-Distill-Qwen-14B-Q3_K_M.gguf" in q3_filenames

    llama_q3 = next(c for c in q3_models if "Meta-Llama-3.1-8B" in c.repo_id)
    assert llama_q3.size_mb == 3740.0
    assert llama_q3.benchmarks["eq_bench"] == 74.8
    assert llama_q3.measured_benchmarks == ["eq_bench", "mt_bench_writing", "arena_elo"]

    r1_14b = next(c for c in q3_models if "DeepSeek-R1-Distill-Qwen-14B" in c.repo_id)
    assert r1_14b.size_mb == 6980.0
    assert r1_14b.benchmarks["gsm8k"] == 93.9
    assert r1_14b.num_layers == 48


def test_load_benchmarks_cache_missing_file():
    """Verify that loading from nonexistent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_benchmarks_cache(Path("nonexistent_path/benchmarks.json"))


def test_normalize_benchmarks_imputation():
    """Verify two-way imputation between SWE-bench/HumanEval and Arena-Elo/OpenLLM."""
    # 1. Impute SWE-bench from HumanEval (80.0 * 0.45 = 36.0) for coder model
    score_swe_imputed = normalize_benchmarks({"humaneval": 80.0}, task="coding", is_coder_model=True)
    # Formula for coding: 0.60 * (36.0 / 50.0) + 0.40 * (80.0 / 100.0) = 0.60 * 0.72 + 0.40 * 0.80 = 0.432 + 0.32 = 0.752
    assert pytest.approx(score_swe_imputed, abs=1e-3) == 0.752

    # 2. Impute HumanEval from SWE-bench (36.0 / 0.45 = 80.0) for coder model
    score_he_imputed = normalize_benchmarks({"swe_bench_verified": 36.0}, task="coding", is_coder_model=True)
    assert pytest.approx(score_he_imputed, abs=1e-3) == 0.752

    # 3. Impute Arena Elo from OpenLLM v2 (60.0 -> 1000 + 3.5 * 60 = 1210.0)
    score_elo_imputed = normalize_benchmarks({"openllm_v2": 60.0}, task="reasoning")
    # Reasoning formula: 0.60 * (60.0 / 100.0) + 0.40 * ((1210.0 - 1000.0) / 350.0)
    # = 0.36 + 0.40 * (210.0 / 350.0) = 0.36 + 0.40 * 0.60 = 0.36 + 0.24 = 0.60
    assert pytest.approx(score_elo_imputed, abs=1e-3) == 0.60

    # 4. Impute OpenLLM v2 from Arena Elo (1210.0 -> (1210 - 1000) / 3.5 = 60.0)
    score_ollm_imputed = normalize_benchmarks({"arena_elo": 1210.0}, task="reasoning")
    assert pytest.approx(score_ollm_imputed, abs=1e-3) == 0.60

    # 5. Empty dictionary safely imputes default baselines
    score_empty = normalize_benchmarks({})
    assert 0.0 <= score_empty <= 1.0


def test_normalize_benchmarks_weights_and_bounds():
    """Verify BenchNorm 4-term weighted default and clamping bounds."""
    # Standard Qwen 7B benchmarks
    benchmarks = {
        "swe_bench_verified": 39.8,
        "humaneval": 88.4,
        "openllm_v2": 67.2,
        "arena_elo": 1220.0,
    }
    score_default = normalize_benchmarks(benchmarks)
    assert 0.0 <= score_default <= 1.0

    # Expected calculation:
    # 0.35 * (39.8 / 50.0) = 0.2786
    # 0.30 * (88.4 / 100.0) = 0.2652
    # 0.20 * (67.2 / 100.0) = 0.1344
    # 0.15 * ((1220 - 1000) / 400.0) = 0.15 * 0.55 = 0.0825
    # Total = 0.2786 + 0.2652 + 0.1344 + 0.0825 = 0.7607
    assert pytest.approx(score_default, abs=1e-2) == 0.76

    # Test clamping: extreme high values clamped to 1.0
    extreme_high = {
        "swe_bench_verified": 200.0,
        "humaneval": 200.0,
        "openllm_v2": 200.0,
        "arena_elo": 3000.0,
    }
    assert normalize_benchmarks(extreme_high) == 1.0

    # Test clamping: negative values clamped to 0.0
    extreme_low = {
        "swe_bench_verified": -50.0,
        "humaneval": -50.0,
        "openllm_v2": -50.0,
        "arena_elo": 500.0,
    }
    assert normalize_benchmarks(extreme_low) == 0.0


def test_normalize_benchmarks_task_domains():
    """Verify task domain selections (coding, reasoning, chat)."""
    benchmarks = {
        "swe_bench_verified": 40.0,
        "humaneval": 80.0,
        "openllm_v2": 70.0,
        "arena_elo": 1250.0,
        "mmlu_pro": 65.0,
    }
    c_score = normalize_benchmarks(benchmarks, task="coding")
    r_score = normalize_benchmarks(benchmarks, task="reasoning")
    ch_score = normalize_benchmarks(benchmarks, task="chat")

    assert 0.0 <= c_score <= 1.0
    assert 0.0 <= r_score <= 1.0
    assert 0.0 <= ch_score <= 1.0


def test_hf_client_offline_mode():
    """Verify get_model_candidates with offline=True bypasses network."""
    candidates = get_model_candidates(offline=True)
    assert len(candidates) >= 7
    assert all(isinstance(c, CandidateModel) for c in candidates)


@patch("companion_packager.research.hf_client.load_benchmarks_cache")
def test_hf_client_online_lfs_size_update(mock_cache):
    """Verify that online query successfully refreshes size_mb from LFS sibling metadata."""
    sample_candidate = CandidateModel(
        repo_id="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        filename="qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        quant_method="Q4_K_M",
        param_count_billions=7.61,
        size_mb=4680.0,
    )
    mock_cache.return_value = [sample_candidate]

    # Mock HfApi
    mock_sibling = MagicMock()
    mock_sibling.rfilename = "qwen2.5-coder-7b-instruct-q4_k_m.gguf"
    mock_sibling.lfs = MagicMock(size=5242880000)  # 5000 MB

    mock_info = MagicMock()
    mock_info.siblings = [mock_sibling]

    with patch("huggingface_hub.HfApi") as mock_hf_api_cls:
        mock_api = MagicMock()
        mock_api.model_info.return_value = mock_info
        mock_hf_api_cls.return_value = mock_api

        candidates = get_model_candidates(offline=False)
        assert len(candidates) == 1
        # 5242880000 / (1024*1024) = 5000.0 MB
        assert pytest.approx(candidates[0].size_mb, abs=0.1) == 5000.0


@patch("companion_packager.research.hf_client.load_benchmarks_cache")
def test_hf_client_gated_and_network_error_fallback(mock_cache):
    """Verify that gated repos (401/403) or network timeouts fall back safely to cached data."""
    sample_candidate = CandidateModel(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        filename="llama-3.1-8b-instruct-q4_k_m.gguf",
        quant_method="Q4_K_M",
        param_count_billions=8.03,
        size_mb=4900.0,
    )
    mock_cache.return_value = [sample_candidate]

    with patch("huggingface_hub.HfApi") as mock_hf_api_cls:
        mock_api = MagicMock()
        # Raise generic exception / connection error
        mock_api.model_info.side_effect = ConnectionError("Could not reach huggingface.co")
        mock_hf_api_cls.return_value = mock_api

        candidates = get_model_candidates(offline=False)
        assert len(candidates) == 1
        # Retains original cached size
        assert candidates[0].size_mb == 4900.0


def test_generate_citation_card():
    """Verify markdown citation card formatting with and without sizing evaluation."""
    candidate = CandidateModel(
        repo_id="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        filename="qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        quant_method="Q4_K_M",
        param_count_billions=7.61,
        size_mb=4680.0,
        citation_doi="10.48550/arXiv.2409.12186",
        paper_url="https://arxiv.org/abs/2409.12186",
        benchmarks={"swe_bench_verified": 39.8, "humaneval": 88.4},
    )

    card_simple = generate_citation_card(candidate)
    assert "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF" in card_simple
    assert "10.48550/arXiv.2409.12186" in card_simple
    assert "https://arxiv.org/abs/2409.12186" in card_simple
    assert "SWE-bench Verified" in card_simple or "Swe Bench Verified" in card_simple

    eval_obj = SizingEvaluation(
        fits=True,
        weights_mb=4680.0,
        kv_cache_mb=512.0,
        overhead_mb=256.0,
        total_ram_mb=5448.0,
        budget_ceiling_mb=6144.0,
        tokens_per_second=18.5,
        ram_penalty=0.0,
    )
    card_with_eval = generate_citation_card(candidate, eval_obj)
    assert "Fits Budget" in card_with_eval
    assert "5448.0 MB" in card_with_eval
    assert "18.5 tokens/sec" in card_with_eval


def test_generate_recommendations_markdown():
    """Verify comprehensive RECOMMENDATIONS.md report generation."""
    hardware = HardwareProfile(
        os_name="Windows 11",
        cpu_brand="11th Gen Intel Core i7",
        cpu_arch="x86_64",
        physical_cores=4,
        logical_cores=8,
        has_avx512=True,
        has_avx2=True,
        has_neon=False,
        ram_total_mb=16384.0,
        ram_avail_mb=12288.0,
        disk_free_gb=100.0,
        bandwidth_gb_s=30.0,
        backend_target="cpu",
        gpus=[
            GPUInfo(
                index=0,
                name="Intel Iris Xe",
                vendor="intel",
                vram_total_mb=1024.0,
                vram_free_mb=1024.0,
            )
        ],
    )

    user_needs = UserNeeds(
        task="coding",
        priority="balanced",
        min_context_tokens=16384,
        require_permissive_license=True,
    )

    candidate = CandidateModel(
        repo_id="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        filename="qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        quant_method="Q4_K_M",
        param_count_billions=7.61,
        size_mb=4680.0,
        citation_doi="10.48550/arXiv.2409.12186",
        paper_url="https://arxiv.org/abs/2409.12186",
        benchmarks={"swe_bench_verified": 39.8, "humaneval": 88.4},
    )

    eval_obj = SizingEvaluation(
        fits=True,
        weights_mb=4680.0,
        kv_cache_mb=512.0,
        overhead_mb=256.0,
        total_ram_mb=5448.0,
        budget_ceiling_mb=6144.0,
        tokens_per_second=18.5,
        ram_penalty=0.0,
    )

    scorecard = Scorecard(
        candidate=candidate,
        evaluation=eval_obj,
        task_fit_score=0.76,
        latency_score=0.85,
        final_score=0.79,
        rank=1,
    )

    md = generate_recommendations_markdown([scorecard], hardware, user_needs)
    assert "# Agent Companion Model Recommendations" in md
    assert "Host Hardware Profile" in md
    assert "Windows 11" in md
    assert "Intel Iris Xe" in md
    assert "User Workload & Operational Constraints" in md
    assert "Primary Recommended Model" in md
    assert "Quickstart Execution Guide" in md
    assert "llama-server" in md
    assert "Academic References & Citations" in md
