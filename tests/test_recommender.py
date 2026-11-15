"""Unit tests for needs questionnaire, candidate ranking, and recommendation engine."""

from typing import List
import pytest
from rich.console import Console

from companion_packager.interview.questionnaire import ask_user_needs
from companion_packager.research.benchmarks import load_benchmarks_cache
from companion_packager.schemas import (
    CandidateModel,
    GPUInfo,
    HardwareProfile,
    Scorecard,
    UserNeeds,
)
from companion_packager.sizing.recommender import (
    PERMISSIVE_LICENSES,
    display_scorecard_table,
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
    assert sc_14b.final_score < -900.0

    assert sc_32b.evaluation.fits is False
    assert sc_32b.evaluation.ram_penalty == 1000.0
    assert sc_32b.final_score < -900.0


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
    assert deepseek_sc.rank <= 2


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
    assert top_sc.final_score > 0.90


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
    assert sc.final_score < -900.0


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
