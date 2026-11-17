"""Unit tests for needs questionnaire, candidate ranking, and recommendation engine."""

from typing import List
import pytest
from rich.console import Console

from companion_packager.interview.questionnaire import ask_user_needs
from companion_packager.research.benchmarks import load_benchmarks_cache
from companion_packager.schemas import (
    CandidateModel,
    ConcurrencyScenario,
    ConcurrencyScenarioBudget,
    GPUInfo,
    HardwareProfile,
    ProcessMemorySummary,
    Scorecard,
    UserNeeds,
)
from companion_packager.sizing.recommender import (
    PERMISSIVE_LICENSES,
    display_concurrency_scenarios_table,
    display_scorecard_table,
    evaluate_all_concurrency_scenarios,
    rank_candidates,
)


def test_ask_user_needs_headless_defaults():
    """Verify ask_user_needs in headless mode produces correct defaults."""
    needs = ask_user_needs(interactive=False)
    assert needs.task == "coding"
    assert needs.priority == "balanced"
    assert needs.min_context_tokens == 16384
    assert needs.require_permissive_license is False


def test_ask_user_needs_headless_custom():
    """Verify ask_user_needs in headless mode respects explicit parameters."""
    needs = ask_user_needs(
        interactive=False,
        task="reasoning",
        priority="speed",
        min_context_tokens=32768,
        require_permissive_license=True,
    )
    assert needs.task == "reasoning"
    assert needs.priority == "speed"
    assert needs.min_context_tokens == 32768
    assert needs.require_permissive_license is True


def test_ask_user_needs_validation_errors():
    """Verify ask_user_needs rejects invalid task, priority, or non-positive context."""
    with pytest.raises(ValueError, match="Invalid task"):
        ask_user_needs(interactive=False, task="invalid_task")

    with pytest.raises(ValueError, match="Invalid priority"):
        ask_user_needs(interactive=False, priority="super_fast")

    with pytest.raises(ValueError, match="min_context_tokens must be positive"):
        ask_user_needs(interactive=False, min_context_tokens=0)


def test_ask_user_needs_interactive_mocked(monkeypatch):
    """Verify ask_user_needs in interactive mode prompts and captures responses."""
    prompts_asked = []

    def mock_prompt_ask(prompt_text, choices=None, default=None):
        prompts_asked.append(str(prompt_text))
        if "task" in str(prompt_text).lower():
            return "reasoning"
        if "optimization" in str(prompt_text).lower() or "priority" in str(prompt_text).lower():
            return "quality"
        if "context" in str(prompt_text).lower():
            return "32768"
        return default

    def mock_confirm_ask(prompt_text, default=False):
        prompts_asked.append(str(prompt_text))
        return True

    monkeypatch.setattr("rich.prompt.Prompt.ask", mock_prompt_ask)
    monkeypatch.setattr("rich.prompt.Confirm.ask", mock_confirm_ask)

    needs = ask_user_needs(interactive=True)
    assert needs.task == "reasoning"
    assert needs.priority == "quality"
    assert needs.min_context_tokens == 32768
    assert needs.require_permissive_license is True
    assert len(prompts_asked) == 4


def test_permissive_license_filtering():
    """Verify permissive license filter excludes proprietary/gated licenses like DeepSeek-license."""
    candidates = load_benchmarks_cache()
    # Check that DeepSeek is in cache with deepseek-license
    deepseek_models = [c for c in candidates if "deepseek" in c.license.lower()]
    assert len(deepseek_models) >= 1

    hw = HardwareProfile(
        os_name="windows",
        cpu_brand="AMD Ryzen 7",
        cpu_arch="x86_64",
        physical_cores=8,
        logical_cores=16,
        has_avx512=True,
        has_avx2=True,
        has_neon=False,
        ram_total_mb=32768.0,
        ram_avail_mb=20000.0,
        disk_free_gb=200.0,
        gpus=[
            GPUInfo(
                index=0,
                name="NVIDIA RTX 4070 Ti",
                vendor="nvidia",
                vram_total_mb=16384.0,
                vram_free_mb=15000.0,
            )
        ],
        bandwidth_gb_s=360.0,
        backend_target="cuda",
    )

    # 1. With require_permissive_license = True -> DeepSeek filtered out
    needs_permissive = UserNeeds(task="coding", priority="balanced", require_permissive_license=True)
    scorecards_permissive = rank_candidates(candidates, hw, needs_permissive)
    for sc in scorecards_permissive:
        assert sc.candidate.license.lower() in PERMISSIVE_LICENSES
        assert "deepseek" not in sc.candidate.license.lower()

    # 2. With require_permissive_license = False -> DeepSeek included
    needs_any = UserNeeds(task="coding", priority="balanced", require_permissive_license=False)
    scorecards_any = rank_candidates(candidates, hw, needs_any)
    repos = [sc.candidate.repo_id for sc in scorecards_any]
    assert any("DeepSeek" in r for r in repos)


def test_cpu_laptop_profile_recommendation():
    """Verify sub-6GB RAM invariant: Qwen2.5-Coder 7B fits under 6144 MB ceiling and ranks #1 among [7B, 14B, 32B]."""
    all_candidates = load_benchmarks_cache()
    cand_7b = next(c for c in all_candidates if c.param_count_billions == 7.61)
    cand_14b = next(c for c in all_candidates if c.param_count_billions == 14.7)
    cand_32b = next(c for c in all_candidates if c.param_count_billions == 32.5)

    # Sub-6GB CPU laptop (8GB RAM, available 5000MB, CPU execution)
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
        ram_avail_mb=8000.0,  # 8000 * 0.80 = 6400 -> capped at 6144.0 MB invariant
        disk_free_gb=100.0,
        gpus=[],
        bandwidth_gb_s=30.0,
        backend_target="cpu",
    )

    user_needs = UserNeeds(task="coding", priority="quality", min_context_tokens=16384)
    scorecards = rank_candidates([cand_7b, cand_14b, cand_32b], laptop_hw, user_needs)

    assert len(scorecards) == 3
    # Qwen2.5-Coder 7B fits under the 6144 MB ceiling
    assert scorecards[0].candidate.repo_id == "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"
    assert scorecards[0].evaluation.fits is True
    assert scorecards[0].evaluation.budget_ceiling_mb <= 6144.0
    assert scorecards[0].rank == 1

    # 14B and 32B are rejected due to exceeding budget ceiling
    sc_14b = next(sc for sc in scorecards if sc.candidate.param_count_billions == 14.7)
    sc_32b = next(sc for sc in scorecards if sc.candidate.param_count_billions == 32.5)

    assert sc_14b.evaluation.fits is False
    assert sc_14b.evaluation.ram_penalty == 1000.0
    assert sc_14b.final_score < -100.0

    assert sc_32b.evaluation.fits is False
    assert sc_32b.evaluation.ram_penalty == 1000.0
    assert sc_32b.final_score < -100.0


def test_mid_tier_gpu_recommendation():
    """Verify DeepSeek-Coder-V2-Lite MoE ranks high due to 0.15 active_weight_ratio throughput speedup."""
    candidates = load_benchmarks_cache()
    # 16GB VRAM mid-tier GPU (e.g. RTX 4070 Ti / 4060 Ti 16GB)
    gpu_hw = HardwareProfile(
        os_name="windows",
        cpu_brand="AMD Ryzen 7 7800X3D",
        cpu_arch="x86_64",
        physical_cores=8,
        logical_cores=16,
        has_avx512=True,
        has_avx2=True,
        has_neon=False,
        ram_total_mb=32768.0,
        ram_avail_mb=24000.0,
        disk_free_gb=300.0,
        gpus=[
            GPUInfo(
                index=0,
                name="NVIDIA GeForce RTX 4070 Ti",
                vendor="nvidia",
                vram_total_mb=16384.0,
                vram_free_mb=15000.0,
            )
        ],
        bandwidth_gb_s=360.0,
        backend_target="cuda",
    )

    user_needs = UserNeeds(task="coding", priority="balanced", min_context_tokens=16384)
    scorecards = rank_candidates(candidates, gpu_hw, user_needs)

    deepseek_sc = next(
        sc for sc in scorecards
        if "DeepSeek-Coder-V2-Lite" in sc.candidate.repo_id
    )
    # MoE 0.15 active weight ratio produces high throughput (> 200 tok/s on 360 GB/s bus)
    assert deepseek_sc.candidate.active_weight_ratio == 0.15
    assert deepseek_sc.evaluation.tokens_per_second > 200.0
    assert deepseek_sc.latency_score == 1.0
    assert deepseek_sc.evaluation.fits is True
    # Ranks in the top tier of recommendations
    assert deepseek_sc.rank <= 3


def test_high_tier_24gb_gpu_recommendation():
    """Verify high-tier GPU workstation recommendation ranks Qwen2.5-Coder 32B #1 for quality."""
    candidates = load_benchmarks_cache()
    # High-tier workstation GPU envelope (24GB+ / 40GB VRAM)
    workstation_hw = HardwareProfile(
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
                name="NVIDIA RTX 4090 Workstation",
                vendor="nvidia",
                vram_total_mb=40000.0,
                vram_free_mb=38000.0,
            )
        ],
        bandwidth_gb_s=1008.0,
        backend_target="cuda",
    )

    # Coding task with quality priority
    user_needs = UserNeeds(task="coding", priority="quality", min_context_tokens=16384)
    scorecards = rank_candidates(candidates, workstation_hw, user_needs)

    # Qwen2.5-Coder 32B achieves the highest quality score and ranks #1
    top_sc = scorecards[0]
    assert top_sc.candidate.repo_id == "Qwen/Qwen2.5-Coder-32B-Instruct-GGUF"
    assert top_sc.evaluation.fits is True
    assert top_sc.rank == 1
    assert top_sc.task_fit_score > 0.95
    assert top_sc.final_score >= 0.85


def test_priority_weighting_shifts():
    """Verify shift between 'quality' and 'speed' priority adjusts rankings and weighting."""
    candidates = load_benchmarks_cache()
    hw = HardwareProfile(
        os_name="windows",
        cpu_brand="AMD Ryzen 7",
        cpu_arch="x86_64",
        physical_cores=8,
        logical_cores=16,
        has_avx512=True,
        has_avx2=True,
        has_neon=False,
        ram_total_mb=32768.0,
        ram_avail_mb=20000.0,
        disk_free_gb=200.0,
        gpus=[
            GPUInfo(
                index=0,
                name="NVIDIA RTX 4070 Ti",
                vendor="nvidia",
                vram_total_mb=16384.0,
                vram_free_mb=15000.0,
            )
        ],
        bandwidth_gb_s=360.0,
        backend_target="cuda",
    )

    needs_quality = UserNeeds(task="coding", priority="quality", min_context_tokens=16384)
    needs_speed = UserNeeds(task="coding", priority="speed", min_context_tokens=16384)

    scores_quality = {sc.candidate.repo_id: sc.final_score for sc in rank_candidates(candidates, hw, needs_quality)}
    scores_speed = {sc.candidate.repo_id: sc.final_score for sc in rank_candidates(candidates, hw, needs_speed)}

    # High benchmark model (14B) vs lightweight fast model (1.5B)
    cand_14b_id = "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF"
    cand_1_5b_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF"

    # Under quality priority, 14B has a higher relative score advantage over 1.5B than under speed priority
    quality_diff = scores_quality[cand_14b_id] - scores_quality[cand_1_5b_id]
    speed_diff = scores_speed[cand_14b_id] - scores_speed[cand_1_5b_id]
    assert quality_diff > speed_diff


def test_context_window_violation_penalized():
    """Verify models with context window smaller than requested are rejected with heavy penalty."""
    cand = CandidateModel(
        repo_id="test/small-context-model",
        filename="model.gguf",
        quant_method="Q4_K_M",
        param_count_billions=3.0,
        size_mb=2000.0,
        context_window=8192,  # Only 8k context supported
        num_layers=28,
        num_kv_heads=4,
        head_dim=128,
    )
    hw = HardwareProfile(
        os_name="windows",
        cpu_brand="Intel",
        cpu_arch="x86_64",
        physical_cores=4,
        logical_cores=8,
        has_avx512=False,
        has_avx2=True,
        has_neon=False,
        ram_total_mb=16384.0,
        ram_avail_mb=12000.0,
        disk_free_gb=100.0,
        gpus=[],
        backend_target="cpu",
    )

    needs = UserNeeds(task="coding", min_context_tokens=16384)  # Requests 16k context
    scorecards = rank_candidates([cand], hw, needs)

    assert len(scorecards) == 1
    sc = scorecards[0]
    assert sc.evaluation.fits is False
    assert sc.evaluation.ram_penalty == 1000.0
    assert "Context window (8192) < requested (16384)" in sc.evaluation.rejection_reason
    assert sc.final_score < -100.0


def test_display_scorecard_table():
    """Verify display_scorecard_table prints without errors to Rich console."""
    candidates = load_benchmarks_cache()[:3]
    hw = HardwareProfile(
        os_name="windows",
        cpu_brand="Intel",
        cpu_arch="x86_64",
        physical_cores=4,
        logical_cores=8,
        has_avx512=False,
        has_avx2=True,
        has_neon=False,
        ram_total_mb=16384.0,
        ram_avail_mb=8000.0,
        disk_free_gb=100.0,
        gpus=[],
        backend_target="cpu",
    )
    needs = UserNeeds(task="coding", priority="balanced")
    scorecards = rank_candidates(candidates, hw, needs)

    console = Console(record=True, width=120)
    display_scorecard_table(scorecards, console=console)
    output = console.export_text()

    assert "Candidate Model Sizing & Recommendation Matrix" in output
    assert "Qwen2.5-Coder" in output
    assert "Rank" in output
    assert "Quant" in output


def test_evaluate_all_concurrency_scenarios_cpu_and_gpu():
    """Verify evaluate_all_concurrency_scenarios returns all 3 scenarios on CPU and GPU profiles."""
    candidates = load_benchmarks_cache()

    # CPU Profile
    cpu_hw = HardwareProfile(
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
        backend_target="cpu",
    )
    needs = UserNeeds(task="coding", priority="balanced")
    ps = ProcessMemorySummary(browsers_mb=3000.0, ides_mb=1500.0, dev_runtimes_mb=500.0, system_mb=1000.0, other_mb=500.0, total_scanned_mb=6500.0)

    cpu_scenarios = evaluate_all_concurrency_scenarios(cpu_hw, needs, candidates, process_summary=ps)
    assert len(cpu_scenarios) == 3
    assert cpu_scenarios[0].scenario == ConcurrencyScenario.PARALLEL
    assert cpu_scenarios[1].scenario == ConcurrencyScenario.MINIMAL_INTERFERENCE
    assert cpu_scenarios[2].scenario == ConcurrencyScenario.DEDICATED
    for sc in cpu_scenarios:
        assert sc.budget_ceiling_mb <= 6144.0
        assert sc.recommended_scorecard is not None
        assert sc.recommended_scorecard.evaluation.fits is True

    # GPU Profile
    gpu_hw = HardwareProfile(
        os_name="windows",
        cpu_brand="AMD Ryzen 9",
        cpu_arch="x86_64",
        physical_cores=16,
        logical_cores=32,
        has_avx512=True,
        has_avx2=True,
        has_neon=False,
        ram_total_mb=65536.0,
        ram_avail_mb=48000.0,
        disk_free_gb=500.0,
        gpus=[
            GPUInfo(
                index=0,
                name="NVIDIA RTX 4090",
                vendor="nvidia",
                vram_total_mb=24576.0,
                vram_free_mb=22000.0,
            )
        ],
        backend_target="cuda",
    )
    gpu_scenarios = evaluate_all_concurrency_scenarios(gpu_hw, needs, candidates, process_summary=ps)
    assert len(gpu_scenarios) == 3
    for sc in gpu_scenarios:
        assert sc.recommended_scorecard is not None
        assert sc.recommended_scorecard.evaluation.fits is True


def test_16gb_cpu_dedicated_wins_7b_8b_colibri_across_tasks():
    """Verify on 16GB CPU host in Dedicated mode: 7B/8B/Colibri models win across Creative Writing, Summarization, and Coding."""
    candidates = load_benchmarks_cache()
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
        backend_target="cpu",
    )
    ps = ProcessMemorySummary(browsers_mb=4000.0, ides_mb=2000.0, dev_runtimes_mb=1000.0, system_mb=1000.0, other_mb=500.0, total_scanned_mb=8500.0)

    # 1. Coding
    needs_coding = UserNeeds(task="coding", priority="balanced")
    scenarios_coding = evaluate_all_concurrency_scenarios(hw, needs_coding, candidates, process_summary=ps)
    dedicated_coding = scenarios_coding[2]
    assert dedicated_coding.scenario == ConcurrencyScenario.DEDICATED
    assert dedicated_coding.recommended_scorecard is not None
    top_coding_repo = dedicated_coding.recommended_scorecard.candidate.repo_id.lower()
    # Wins with 7B or 27B Colibri model
    assert "qwen2.5-coder-7b" in top_coding_repo or "ternaryquench" in top_coding_repo
    assert dedicated_coding.recommended_scorecard.candidate.param_count_billions >= 7.0

    # 2. Creative Writing
    needs_writing = UserNeeds(task="creative_writing", priority="balanced")
    scenarios_writing = evaluate_all_concurrency_scenarios(hw, needs_writing, candidates, process_summary=ps)
    dedicated_writing = scenarios_writing[2]
    assert dedicated_writing.recommended_scorecard is not None
    top_writing_repo = dedicated_writing.recommended_scorecard.candidate.repo_id.lower()
    # Wins with 8B Llama-3.1 or 7B Mistral prose model
    assert "llama-3.1-8b" in top_writing_repo or "mistral" in top_writing_repo
    assert dedicated_writing.recommended_scorecard.candidate.param_count_billions >= 7.0

    # 3. Summarization
    needs_summary = UserNeeds(task="summarization", priority="balanced")
    scenarios_summary = evaluate_all_concurrency_scenarios(hw, needs_summary, candidates, process_summary=ps)
    dedicated_summary = scenarios_summary[2]
    assert dedicated_summary.recommended_scorecard is not None
    top_summary_repo = dedicated_summary.recommended_scorecard.candidate.repo_id.lower()
    # Wins with Qwen2.5-7B
    assert "qwen2.5-7b" in top_summary_repo
    assert dedicated_summary.recommended_scorecard.candidate.param_count_billions >= 7.0


def test_tight_parallel_mode_1_5b_3b_wins():
    """Verify that in tight Parallel mode (2.5GB free): 1.5B/3B models win to prevent interference."""
    candidates = load_benchmarks_cache()
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
        ram_avail_mb=2500.0,  # Tight available RAM
        disk_free_gb=100.0,
        gpus=[],
        backend_target="cpu",
    )
    ps = ProcessMemorySummary(browsers_mb=6000.0, ides_mb=3000.0, dev_runtimes_mb=2000.0, system_mb=1500.0, other_mb=1000.0, total_scanned_mb=13500.0)

    needs = UserNeeds(task="coding", priority="balanced")
    scenarios = evaluate_all_concurrency_scenarios(hw, needs, candidates, process_summary=ps)
    parallel = scenarios[0]
    assert parallel.scenario == ConcurrencyScenario.PARALLEL
    # In tight parallel mode with 2500 MB avail * 0.75 = 1875 MB budget ceiling, only 1.5B/3B fits
    assert parallel.recommended_scorecard is not None
    assert parallel.recommended_scorecard.evaluation.fits is True
    assert parallel.recommended_scorecard.candidate.param_count_billions <= 3.5
    assert "1.5b" in parallel.recommended_scorecard.candidate.repo_id.lower() or "3b" in parallel.recommended_scorecard.candidate.repo_id.lower() or "smollm" in parallel.recommended_scorecard.candidate.repo_id.lower()


def test_display_concurrency_scenarios_table_normal_and_none():
    """Verify display_concurrency_scenarios_table renders cleanly with normal and None scorecards."""
    ps = ProcessMemorySummary(browsers_mb=2048.0, ides_mb=1024.0, dev_runtimes_mb=512.0, other_mb=256.0, total_scanned_mb=3840.0)

    cand = CandidateModel(
        repo_id="Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
        filename="qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
        quant_method="Q4_K_M",
        param_count_billions=1.54,
        size_mb=1150.0,
    )
    from companion_packager.schemas import SizingEvaluation, ProofCitation
    eval_res = SizingEvaluation(
        fits=True,
        weights_mb=985.6,
        kv_cache_mb=128.0,
        overhead_mb=400.0,
        total_ram_mb=1513.6,
        budget_ceiling_mb=3000.0,
        tokens_per_second=22.5,
    )
    proof_cit = ProofCitation(
        model_id=cand.repo_id,
        task_benchmark_name="SWE-bench Verified",
        task_benchmark_score=21.0,
        is_measured=True,
        ram_allocated_mb=1513.6,
        ram_headroom_mb=1486.4,
        tokens_per_second=22.5,
    )
    sc = Scorecard(
        candidate=cand,
        evaluation=eval_res,
        task_fit_score=0.85,
        latency_score=1.0,
        final_score=0.88,
        proof_citation=proof_cit,
    )

    scenarios = [
        ConcurrencyScenarioBudget(
            scenario=ConcurrencyScenario.PARALLEL,
            title="Parallel Coexistence",
            description="Background companion",
            available_ram_mb=4000.0,
            reclaimable_ram_mb=0.0,
            budget_ceiling_mb=3000.0,
            recommended_scorecard=sc,
        ),
        ConcurrencyScenarioBudget(
            scenario=ConcurrencyScenario.MINIMAL_INTERFERENCE,
            title="Minimal Interference",
            description="Reclaim idle browsers",
            available_ram_mb=4000.0,
            reclaimable_ram_mb=1000.0,
            budget_ceiling_mb=4000.0,
            recommended_scorecard=None,  # Tests None branch
        ),
    ]

    console = Console(record=True, width=140)
    display_concurrency_scenarios_table(scenarios, ps, console=console)
    output = console.export_text()

    assert "Operational Concurrency Scenarios & Hardware Sizing" in output
    assert "Active Process RAM: Browsers 2048 MB" in output
    assert "Parallel Coexistence" in output
    assert "Minimal Interference" in output
    assert "Qwen2.5-Coder-1.5B" in output
    assert "SWE-bench Verified: 21.0" in output
    assert "[Measured]" in output
    assert "No candidate fits" in output
