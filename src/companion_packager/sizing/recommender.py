"""Candidate model recommendation and hardware sizing matrix."""

from typing import Dict, List, Optional, Set
from rich.console import Console
from rich.table import Table

from companion_packager.hardware.process_inspector import audit_active_processes
from companion_packager.research.benchmarks import (
    normalize_benchmarks,
    resolve_custom_domain,
)
from companion_packager.research.citations import generate_proof_citation
from companion_packager.schemas import (
    CandidateModel,
    ConcurrencyScenario,
    ConcurrencyScenarioBudget,
    HardwareProfile,
    ProcessMemorySummary,
    Scorecard,
    UserNeeds,
)
from companion_packager.sizing.formulas import (
    calculate_scenario_budgets,
    evaluate_sizing,
)

PERMISSIVE_LICENSES: Set[str] = {
    "apache-2.0",
    "mit",
    "bsd-3-clause",
    "bsd-2-clause",
}


def rank_candidates(
    candidates: List[CandidateModel],
    hardware: HardwareProfile,
    user_needs: UserNeeds,
    sparsity_speedup: float = 0.0,
    hard_exclude: bool = False,
    budget_ceiling_override_mb: Optional[float] = None,
) -> List[Scorecard]:
    """Score, filter, and rank candidate models against host hardware profile and user needs."""
    # Filter for permissive license if required
    if user_needs.require_permissive_license:
        candidates = [
            c for c in candidates
            if c.license.lower().strip() in PERMISSIVE_LICENSES
        ]

    # Priority weighting profile
    priority = user_needs.priority.lower().strip()
    if priority in ("quality", "max_iq"):
        w_bench = 0.70
        w_speed = 0.15
        w_penalty = 0.15
    elif priority == "speed":
        w_bench = 0.25
        w_speed = 0.60
        w_penalty = 0.15
    else:  # balanced
        w_bench = 0.50
        w_speed = 0.35
        w_penalty = 0.15

    target_device = hardware.backend_target.lower()
    is_cpu = "cpu" in target_device

    # TargetTokSec backend-aware throughput targets
    if priority == "speed":
        target_tok_s = 15.0 if is_cpu else 30.0
    elif priority in ("max_iq", "quality"):
        target_tok_s = 5.0 if is_cpu else 8.0
    else:  # balanced
        target_tok_s = 8.0 if is_cpu else 15.0

    task = user_needs.task.lower().strip() if user_needs.task else "general_desktop"
    effective_task = resolve_custom_domain(user_needs.custom_intent) if task == "custom" else task

    scorecards: List[Scorecard] = []

    for candidate in candidates:
        evaluation = evaluate_sizing(
            candidate=candidate,
            hardware=hardware,
            context_tokens=user_needs.min_context_tokens,
            bits_per_weight=4.5,
            sparsity_speedup=sparsity_speedup,
            budget_ceiling_override_mb=budget_ceiling_override_mb,
        )

        # Hard-exclusion if weights exceed budget ceiling and hard_exclude requested
        if hard_exclude and evaluation.weights_mb > evaluation.budget_ceiling_mb:
            continue

        # Context length verification
        if candidate.context_window < user_needs.min_context_tokens:
            evaluation = evaluation.model_copy(
                update={
                    "fits": False,
                    "ram_penalty": 1000.0,
                    "rejection_reason": (
                        f"Context window ({candidate.context_window}) < "
                        f"requested ({user_needs.min_context_tokens})"
                    ),
                }
            )

        repo = candidate.repo_id.lower()
        is_code_model = "coder" in repo or "code" in repo or "ternaryquench" in repo
        is_prose_model = ("llama-3.1" in repo or "mistral" in repo) and "llama-3.2-3b" not in repo
        is_summary_model = "qwen2.5" in repo and "coder" not in repo
        is_math_model = "r1" in repo or "math" in repo
        is_desktop_model = (
            "general" in repo
            or "smollm" in repo
            or "llama-3.2-3b" in repo
            or ("qwen2.5-7b" in repo and "coder" not in repo and effective_task in ("general_desktop", "custom"))
        ) and not (is_code_model or is_math_model)

        bench_norm = normalize_benchmarks(
            candidate.benchmarks,
            task=user_needs.task,
            is_coder_model=is_code_model,
            is_math_model=is_math_model,
            is_prose_model=is_prose_model,
            is_summary_model=is_summary_model,
            custom_intent=user_needs.custom_intent,
        )

        # Domain affinity bonus
        if effective_task == "coding" and is_code_model:
            domain_bonus = 0.10
        elif effective_task == "creative_writing" and is_prose_model:
            domain_bonus = 0.10
        elif effective_task == "general_desktop" and is_desktop_model:
            domain_bonus = 0.10
        elif effective_task == "reasoning_math" and is_math_model:
            domain_bonus = 0.10
        elif effective_task == "summarization" and is_summary_model:
            domain_bonus = 0.10
        elif task == "custom" and is_desktop_model:
            domain_bonus = 0.10
        else:
            domain_bonus = 0.0

        # Colibri ternary engine affinity bonus on AVX-512 CPU for matching domain
        is_ternary = (
            candidate.quant_method.upper() in ("TERNARY", "BITNET", "COLIBRI")
            or "ternaryquench" in repo
        )
        if is_ternary and hardware.has_avx512 and is_cpu and (domain_bonus > 0.0 or (is_code_model and effective_task in ("coding", "custom"))):
            engine_bonus = 0.15
        else:
            engine_bonus = 0.0

        task_fit = min(1.0, bench_norm + domain_bonus + engine_bonus)
        speed_score = min(1.0, max(0.0, evaluation.tokens_per_second / target_tok_s))
        final_score = (w_bench * task_fit + w_speed * speed_score) - (w_penalty * evaluation.ram_penalty)

        proof_cit = generate_proof_citation(
            candidate=candidate,
            task=user_needs.task,
            eval_result=evaluation,
            budget_mb=evaluation.budget_ceiling_mb,
            custom_intent=user_needs.custom_intent,
        )

        scorecard = Scorecard(
            candidate=candidate,
            evaluation=evaluation,
            task_fit_score=round(task_fit, 4),
            latency_score=round(speed_score, 4),
            final_score=round(final_score, 4),
            rank=1,
            proof_citation=proof_cit,
        )
        scorecards.append(scorecard)

    # Sort descending by final_score, breaking ties with parameter count
    scorecards.sort(
        key=lambda s: (s.final_score, s.candidate.param_count_billions),
        reverse=True,
    )
    for i, sc in enumerate(scorecards):
        sc.rank = i + 1

    return scorecards


def display_scorecard_table(
    scorecards: List[Scorecard],
    console: Optional[Console] = None,
) -> None:
    """Render Rich terminal scorecard table visualizing candidate rankings."""
    if console is None:
        console = Console()

    table = Table(
        title="[bold green]Candidate Model Sizing & Recommendation Matrix[/bold green]",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Rank", justify="center", style="bold cyan")
    table.add_column("Model", style="bold")
    table.add_column("Quant", justify="center", style="green")
    table.add_column("Est. Speed", justify="right", style="magenta")
    table.add_column("RAM Req", justify="right", style="yellow")
    table.add_column("Budget", justify="right", style="dim")
    table.add_column("Status", justify="center")
    table.add_column("Score", justify="right", style="bold")

    for sc in scorecards:
        status = "[green]FITS[/green]" if sc.evaluation.fits else "[red]EXCEEDS[/red]"
        table.add_row(
            f"#{sc.rank}",
            sc.candidate.repo_id,
            sc.candidate.quant_method,
            f"{sc.evaluation.tokens_per_second:.1f} tok/s",
            f"{sc.evaluation.total_ram_mb:.0f} MB",
            f"{sc.evaluation.budget_ceiling_mb:.0f} MB",
            status,
            f"{sc.final_score:.4f}",
        )

    console.print(table)


def evaluate_all_concurrency_scenarios(
    hardware: HardwareProfile,
    user_needs: UserNeeds,
    candidates: List[CandidateModel],
    process_summary: Optional[ProcessMemorySummary] = None,
) -> List[ConcurrencyScenarioBudget]:
    """Evaluate candidate model feasibility and top recommendation across all 3 concurrency scenarios."""
    if process_summary is None:
        process_summary = audit_active_processes()

    scenario_budgets = calculate_scenario_budgets(hardware, process_summary=process_summary)

    for scenario_budget in scenario_budgets.values():
        scorecards = rank_candidates(
            candidates=candidates,
            hardware=hardware,
            user_needs=user_needs,
            hard_exclude=True,
            budget_ceiling_override_mb=scenario_budget.budget_ceiling_mb,
        )
        valid_scorecards = [s for s in scorecards if s.evaluation.fits]
        if valid_scorecards:
            scenario_budget.recommended_scorecard = valid_scorecards[0]
        else:
            scenario_budget.recommended_scorecard = None

    return [
        scenario_budgets[ConcurrencyScenario.PARALLEL],
        scenario_budgets[ConcurrencyScenario.MINIMAL_INTERFERENCE],
        scenario_budgets[ConcurrencyScenario.DEDICATED],
    ]


def display_concurrency_scenarios_table(
    scenarios: List[ConcurrencyScenarioBudget],
    process_summary: ProcessMemorySummary,
    console: Optional[Console] = None,
) -> None:
    """Render Rich terminal table visualizing concurrency scenario memory partitioning and recommendations."""
    if console is None:
        console = Console()

    console.print(
        f"Active Process RAM: Browsers {process_summary.browsers_mb:.0f} MB | "
        f"IDEs {process_summary.ides_mb:.0f} MB | "
        f"Dev Runtimes {process_summary.dev_runtimes_mb:.0f} MB | "
        f"Other {process_summary.other_mb:.0f} MB | "
        f"Scanned Total {process_summary.total_scanned_mb:.0f} MB"
    )

    table = Table(
        title="[bold green]Operational Concurrency Scenarios & Hardware Sizing[/bold green]",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Scenario", style="bold cyan")
    table.add_column("RAM Budget", justify="right", style="dim")
    table.add_column("Recommended Model", style="bold")
    table.add_column("Quant", justify="center", style="green")
    table.add_column("Est. Speed", justify="right", style="magenta")
    table.add_column("RAM Footprint", justify="right", style="yellow")
    table.add_column("Headroom", justify="right")
    table.add_column("Benchmark / Proof", style="bold")

    for scenario in scenarios:
        if scenario.recommended_scorecard is None:
            table.add_row(
                scenario.title,
                f"{scenario.budget_ceiling_mb:.0f} MB",
                "[red]No candidate fits[/red]",
                "N/A",
                "N/A",
                "N/A",
                "0 MB",
                "[dim]Exceeds available scenario memory[/dim]",
            )
        else:
            sc = scenario.recommended_scorecard
            headroom = scenario.budget_ceiling_mb - sc.evaluation.total_ram_mb
            proof_text = (
                f"{sc.proof_citation.headline_metric_name}: {sc.proof_citation.headline_metric_value} [{sc.proof_citation.headline_metric_tag}]"
                if sc.proof_citation
                else "FITS"
            )
            table.add_row(
                scenario.title,
                f"{scenario.budget_ceiling_mb:.0f} MB",
                sc.candidate.repo_id,
                sc.candidate.quant_method,
                f"{sc.evaluation.tokens_per_second:.1f} tok/s",
                f"{sc.evaluation.total_ram_mb:.0f} MB",
                f"{headroom:.0f} MB",
                proof_text,
            )

    console.print(table)
