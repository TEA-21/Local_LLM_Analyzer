"""Tests for Phase 8: Multi-Domain Intent Query Planner & 3-Pillar Proof Citation Engine."""

import pytest

from companion_packager.interview.questionnaire import (
    ask_user_needs,
    parse_custom_intent,
    VALID_TASKS,
)
from companion_packager.research.benchmarks import (
    calculate_elo_norm,
    clamp,
    impute_missing_benchmarks,
    load_benchmarks_cache,
    normalize_benchmarks,
)
from companion_packager.research.citations import (
    generate_proof_citation,
    generate_proof_citation_card,
    generate_recommendations_markdown,
)
from companion_packager.schemas import (
    CandidateModel,
    HardwareProfile,
    ProofCitation,
    Scorecard,
    SizingEvaluation,
    UserNeeds,
)
from companion_packager.sizing.formulas import estimate_tokens_per_second
from companion_packager.sizing.recommender import rank_candidates


# ==============================================================================
# 1. Custom Intent Tokenizer
# ==============================================================================

def test_parse_custom_intent_languages_and_symbols():
    """Verify tokenizer captures programming languages and symbols like C++, C#, SQL, and RAG."""
    query = "fast c++ and c# compiler"
    tokens = parse_custom_intent(query)
    assert "c++" in tokens
    assert "c#" in tokens
    assert "compiler" in tokens


def test_parse_custom_intent_rag_and_stopwords():
    """Verify stopword stripping and ranking for RAG and analytics queries."""
    query = "the best tool for rag and sql analytics"
    tokens = parse_custom_intent(query)
    # Stopwords 'the', 'best', 'tool', 'for', 'and' should be stripped
    assert "rag" in tokens
    assert "sql" in tokens
    assert "analytics" in tokens
    assert "best" not in tokens
    assert "the" not in tokens


def test_parse_custom_intent_empty_and_fallback():
    """Verify empty or pure stopword queries cleanly fall back to default assistant tokens."""
    assert parse_custom_intent("") == ["instruct", "assistant", "general"]
    assert parse_custom_intent("   ") == ["instruct", "assistant", "general"]
    assert parse_custom_intent("the and for with this that") == ["instruct", "assistant", "general"]


def test_ask_user_needs_multi_domain_and_custom():
    """Verify ask_user_needs accepts multi-domain tasks and custom freeform intent."""
    needs_desktop = ask_user_needs(interactive=False, task="general_desktop")
    assert needs_desktop.task == "general_desktop"

    needs_creative = ask_user_needs(interactive=False, task="creative_writing")
    assert needs_creative.task == "creative_writing"

    needs_custom = ask_user_needs(
        interactive=False,
        task="custom",
        custom_intent="deep rust systems programming",
    )
    assert needs_custom.task == "custom"
    assert needs_custom.custom_intent == "deep rust systems programming"


# ==============================================================================
# 2. Connected Imputation Graph & Bridge
# ==============================================================================

def test_imputation_island_bridge_coding_to_desktop():
    """Verify bidirectional bridge: Island A (SWE/HumanEval) propagates to Island B (IFEval/Arena Elo)."""
    # Start with only HumanEval (Island A)
    initial = {"humaneval": 88.0}
    imputed = impute_missing_benchmarks(initial)

    # Directly measured
    assert imputed["humaneval"] == 88.0

    # 1-hop Island A
    assert pytest.approx(imputed["swe_bench_verified"], abs=1e-1) == 0.45 * 88.0
    # HumanEval -> GSM8K
    expected_gsm = clamp(1.15 * 88.0 - 15.0, 20.0, 98.0)
    assert pytest.approx(imputed["gsm8k"], abs=1e-1) == expected_gsm

    # Bridge: GSM8K -> OpenLLM_v2 (Island B)
    expected_ollm = clamp((expected_gsm - 5.0) / 1.10, 0.0, 100.0)
    assert pytest.approx(imputed["openllm_v2"], abs=1e-1) == expected_ollm

    # OpenLLM_v2 -> Arena_Elo -> IFEval
    expected_elo = clamp(1000.0 + 3.5 * expected_ollm, 950.0, 1350.0)
    assert pytest.approx(imputed["arena_elo"], abs=1e-1) == expected_elo

    expected_ifeval = clamp(0.18 * expected_elo - 130.0, 40.0, 95.0)
    assert pytest.approx(imputed["ifeval"], abs=1e-1) == expected_ifeval


def test_imputation_island_bridge_desktop_to_coding():
    """Verify reverse bridge: Island B (IFEval/Arena Elo) propagates to Island A (HumanEval/SWE)."""
    # Start with only IFEval (Island B)
    initial = {"ifeval": 78.4}
    imputed = impute_missing_benchmarks(initial)

    assert imputed["ifeval"] == 78.4
    expected_elo = clamp((78.4 + 130.0) / 0.18, 950.0, 1350.0)
    assert pytest.approx(imputed["arena_elo"], abs=1e-1) == expected_elo

    # Arena Elo -> OpenLLM_v2
    expected_ollm = clamp((expected_elo - 1000.0) / 3.5, 0.0, 100.0)
    assert pytest.approx(imputed["openllm_v2"], abs=1e-1) == expected_ollm

    # Bridge: OpenLLM_v2 -> GSM8K
    expected_gsm = clamp(1.10 * expected_ollm + 5.0, 20.0, 98.0)
    assert pytest.approx(imputed["gsm8k"], abs=1e-1) == expected_gsm

    # GSM8K -> HumanEval
    expected_he = clamp((expected_gsm + 15.0) / 1.15, 0.0, 100.0)
    assert pytest.approx(imputed["humaneval"], abs=1e-1) == expected_he


def test_imputation_precedence_and_multi_source_mean():
    """Verify direct measured benchmarks are preserved and multiple 1-hop sources are averaged."""
    # IFEval and EQ-Bench both estimate Arena_Elo in 1-hop
    initial = {
        "ifeval": 78.4,
        "eq_bench": 76.5,
    }
    imputed = impute_missing_benchmarks(initial)

    # Directly measured metrics preserved exactly
    assert imputed["ifeval"] == 78.4
    assert imputed["eq_bench"] == 76.5

    est1_elo = clamp((78.4 + 130.0) / 0.18, 950.0, 1350.0)
    est2_elo = clamp((76.5 + 8.0) / 0.07, 950.0, 1350.0)
    expected_mean_elo = (est1_elo + est2_elo) / 2.0

    assert pytest.approx(imputed["arena_elo"], abs=1e-1) == expected_mean_elo


def test_calculate_elo_norm_bounds():
    """Verify standardized Elo normalization function and clamping bounds."""
    assert calculate_elo_norm(950.0) == 0.0
    assert calculate_elo_norm(1350.0) == 1.0
    assert calculate_elo_norm(1150.0) == 0.5
    assert calculate_elo_norm(500.0) == 0.0
    assert calculate_elo_norm(2000.0) == 1.0


# ==============================================================================
# 3. Domain BenchNorm across all 6 domains
# ==============================================================================

def test_domain_benchnorm_formulas():
    """Verify BenchNorm formulas for coding, creative_writing, general_desktop, reasoning_math, summarization, custom."""
    benchmarks = {
        "swe_bench_verified": 40.0,
        "humaneval": 80.0,
        "eq_bench": 75.0,
        "mt_bench_writing": 8.5,
        "ifeval": 80.0,
        "arena_elo": 1250.0,
        "gsm8k": 90.0,
        "math": 70.0,
        "openllm_v2": 70.0,
        "mmlu_pro": 60.0,
    }

    elo_norm = calculate_elo_norm(1250.0)  # (1250 - 950) / 400 = 0.75

    # 1. Coding: 0.60 * (40/50) + 0.40 * (80/100) = 0.60 * 0.80 + 0.40 * 0.80 = 0.80
    c_score = normalize_benchmarks(benchmarks, task="coding")
    assert pytest.approx(c_score, abs=1e-3) == 0.80

    # 2. Creative Writing: 0.50 * (75/100) + 0.30 * (8.5/10) + 0.20 * 0.75 = 0.375 + 0.255 + 0.150 = 0.780
    cw_score = normalize_benchmarks(benchmarks, task="creative_writing")
    assert pytest.approx(cw_score, abs=1e-3) == 0.780

    # 3. General Desktop: 0.60 * (80/100) + 0.40 * 0.75 = 0.48 + 0.30 = 0.780
    gd_score = normalize_benchmarks(benchmarks, task="general_desktop")
    assert pytest.approx(gd_score, abs=1e-3) == 0.780

    # 4. Reasoning Math: 0.50 * (90/100) + 0.30 * (70/100) + 0.20 * (70/100) = 0.45 + 0.21 + 0.14 = 0.800
    rm_score = normalize_benchmarks(benchmarks, task="reasoning_math")
    assert pytest.approx(rm_score, abs=1e-3) == 0.800

    # 5. Summarization: 0.50 * (60/100) + 0.50 * 0.75 = 0.30 + 0.375 = 0.675
    sum_score = normalize_benchmarks(benchmarks, task="summarization")
    assert pytest.approx(sum_score, abs=1e-3) == 0.675

    # 6. Custom: maps to general_desktop composite
    cust_score = normalize_benchmarks(benchmarks, task="custom")
    assert pytest.approx(cust_score, abs=1e-3) == 0.780


# ==============================================================================
# 4. 3-Pillar Proof Citation Engine
# ==============================================================================

def test_proof_citation_generation_and_provenance():
    """Verify ProofCitation generates task-accurate metrics, provenance flags, and bullet justifications."""
    candidate = CandidateModel(
        repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
        filename="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        quant_method="Q4_K_M",
        param_count_billions=3.2,
        size_mb=2050.0,
        benchmarks={"ifeval": 78.4, "arena_elo": 1160.0},
        measured_benchmarks=["ifeval", "arena_elo"],
        proof_justification="Optimized 3B instruction follower with 78.4% IFEval accuracy.",
    )

    evaluation = SizingEvaluation(
        fits=True,
        weights_mb=2050.0,
        kv_cache_mb=400.0,
        overhead_mb=200.0,
        total_ram_mb=2650.0,
        budget_ceiling_mb=6144.0,
        tokens_per_second=24.5,
        ram_penalty=0.0,
    )

    # 1. Desktop task: IFEval is measured
    citation_desktop = generate_proof_citation(
        candidate=candidate,
        task="general_desktop",
        eval_result=evaluation,
        budget_mb=6144.0,
    )
    assert citation_desktop.model_id == "bartowski/Llama-3.2-3B-Instruct-GGUF"
    assert citation_desktop.task_benchmark_name == "IFEval"
    assert citation_desktop.task_benchmark_score == 78.4
    assert citation_desktop.is_measured is True
    assert citation_desktop.ram_allocated_mb == 2650.0
    assert citation_desktop.ram_headroom_mb == 6144.0 - 2650.0
    assert citation_desktop.tokens_per_second == 24.5

    # 2. Citation card markdown output
    card_md = generate_proof_citation_card(citation_desktop)
    assert "3-Pillar Proof Citation" in card_md
    assert "IFEval = `78.4` [Measured]" in card_md
    assert "`2650.0 MB` allocated" in card_md
    assert "Why This Model Justification" in card_md
    assert "Optimized 3B instruction follower" in card_md

    # 3. Coding task: SWE-bench Verified is imputed [Estimated]
    citation_coding = generate_proof_citation(
        candidate=candidate,
        task="coding",
        eval_result=evaluation,
        budget_mb=6144.0,
    )
    assert citation_coding.is_measured is False
    card_coding_md = generate_proof_citation_card(citation_coding)
    assert "[Estimated]" in card_coding_md


# ==============================================================================
# 5. Hardware Ceiling CPU Gating
# ==============================================================================

def test_hardware_ceiling_cpu_gating():
    """Verify that large GPU models (14B, 32B, V2-Lite) are strictly rejected on CPU hosts."""
    candidates = load_benchmarks_cache()
    # 8GB RAM CPU Host -> Budget ceiling is capped by min(8000 * 0.8, 6144) = 6144 MB
    cpu_hw = HardwareProfile(
        os_name="windows",
        cpu_brand="Intel Core i5-1135G7",
        cpu_arch="x86_64",
        physical_cores=4,
        logical_cores=8,
        has_avx512=True,
        has_avx2=True,
        has_neon=False,
        ram_total_mb=8192.0,
        ram_avail_mb=8000.0,
        disk_free_gb=120.0,
        bandwidth_gb_s=30.0,
        backend_target="cpu",
        gpus=[],
    )

    needs = UserNeeds(task="coding", priority="balanced", min_context_tokens=16384)

    # 1. With hard_exclude=True, models with weights > 6144 MB must not appear in recommendations
    filtered_scorecards = rank_candidates(candidates, cpu_hw, needs, hard_exclude=True)
    for sc in filtered_scorecards:
        assert sc.evaluation.weights_mb <= 6144.0
        assert sc.candidate.param_count_billions < 14.0

    # Models exceeding 6144MB (Qwen 14B: 9200MB, Qwen 32B: 19800MB, DeepSeek V2-Lite: 9600MB, Ternary: 10360MB) excluded
    filtered_repos = {sc.candidate.repo_id for sc in filtered_scorecards}
    assert "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF" not in filtered_repos
    assert "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF" not in filtered_repos
    assert "bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF" not in filtered_repos

    # 2. Without hard_exclude, they must have fits=False, ram_penalty=1000.0, and negative final score
    all_scorecards = rank_candidates(candidates, cpu_hw, needs, hard_exclude=False)
    v2_lite_sc = next(sc for sc in all_scorecards if "DeepSeek-Coder-V2-Lite" in sc.candidate.repo_id)
    assert v2_lite_sc.evaluation.fits is False
    assert v2_lite_sc.evaluation.ram_penalty == 1000.0
    assert v2_lite_sc.final_score < -900.0


# ==============================================================================
# 6. Throughput Sizing Formula Sparsity and MoE Clamping
# ==============================================================================

def test_estimate_tokens_per_second_clamping_and_bounds():
    """Verify sparsity factor clamping to [0, 1] and active weight ratio lower-bounding to 0.05."""
    # Active weight ratio clamped to at least 0.05
    tok_s_min_ratio = estimate_tokens_per_second(
        bandwidth_gb_s=300.0,
        weights_mb=9600.0,
        active_weight_ratio=0.001,  # Below 0.05
        sparsity_speedup=0.0,
    )
    expected_effective_ratio = 0.05
    expected_denom = (9600.0 / 1024.0) * expected_effective_ratio
    expected_tok_s = 300.0 / expected_denom
    assert pytest.approx(tok_s_min_ratio, rel=1e-3) == expected_tok_s

    # Sparsity factor clamped to 1.0 maximum
    tok_s_max_sparsity = estimate_tokens_per_second(
        bandwidth_gb_s=300.0,
        weights_mb=9200.0,
        active_weight_ratio=1.0,
        sparsity_factor=2.5,  # Clamped to 1.0
    )
    expected_sparse_tok_s = (300.0 * 2.0) / (9200.0 / 1024.0)
    assert pytest.approx(tok_s_max_sparsity, rel=1e-3) == expected_sparse_tok_s
