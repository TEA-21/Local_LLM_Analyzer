"""Candidate model recommendation and hardware sizing matrix."""

from typing import List, Optional, Set
from rich.console import Console
from rich.table import Table

from companion_packager.research.benchmarks import normalize_benchmarks
from companion_packager.research.citations import generate_proof_citation
from companion_packager.schemas import (
    CandidateModel,
    HardwareProfile,
    Scorecard,
    UserNeeds,
)
from companion_packager.sizing.formulas import evaluate_sizing

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
        w_task = 0.70
        w_speed = 0.30
    elif priority == "speed":
        w_task = 0.30
        w_speed = 0.70
    else:  # balanced
        w_task = 0.50
        w_speed = 0.50

    scorecards: List[Scorecard] = []

    for candidate in candidates:
        evaluation = evaluate_sizing(
            candidate=candidate,
            hardware=hardware,
            context_tokens=user_needs.min_context_tokens,
            bits_per_weight=4.5,
            sparsity_speedup=sparsity_speedup,
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

        task_fit = normalize_benchmarks(candidate.benchmarks, task=user_needs.task)
        speed_score = min(max(evaluation.tokens_per_second / 50.0, 0.0), 1.0)
        final_score = (w_task * task_fit + w_speed * speed_score) - evaluation.ram_penalty

        proof_cit = generate_proof_citation(
            candidate=candidate,
            task=user_needs.task,
            eval_result=evaluation,
            budget_mb=evaluation.budget_ceiling_mb,
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

    # Sort descending by final_score
    scorecards.sort(key=lambda s: s.final_score, reverse=True)
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
