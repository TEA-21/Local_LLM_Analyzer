"""Citation synthesizer, 3-pillar proof generator, and RECOMMENDATIONS.md report generator."""

from datetime import datetime, timezone
from typing import Dict, List, Optional

from companion_packager.research.benchmarks import impute_missing_benchmarks
from companion_packager.schemas import (
    CandidateModel,
    HardwareProfile,
    ProofCitation,
    Scorecard,
    SizingEvaluation,
    UserNeeds,
)

DOMAIN_PRIMARY_BENCHMARKS = {
    "coding": ("swe_bench_verified", "SWE-bench Verified", "humaneval", "HumanEval"),
    "creative_writing": ("eq_bench", "EQ-Bench", "mt_bench_writing", "MT-Bench Writing"),
    "general_desktop": ("ifeval", "IFEval", "arena_elo", "Arena Elo"),
    "reasoning_math": ("gsm8k", "GSM8K", "math", "MATH"),
    "reasoning": ("gsm8k", "GSM8K", "math", "MATH"),
    "summarization": ("mmlu_pro", "MMLU-Pro", "ifeval", "IFEval"),
    "custom": ("ifeval", "IFEval", "arena_elo", "Arena Elo"),
    "chat": ("ifeval", "IFEval", "arena_elo", "Arena Elo"),
}


def generate_proof_citation(
    candidate: CandidateModel,
    task: str,
    eval_result: SizingEvaluation,
    budget_mb: float,
) -> ProofCitation:
    """Generate empirical 3-pillar proof citation for a candidate model."""
    task_clean = task.lower().strip() if task else "coding"
    prim_key, prim_name, alt_key, alt_name = DOMAIN_PRIMARY_BENCHMARKS.get(
        task_clean, ("swe_bench_verified", "SWE-bench Verified", "humaneval", "HumanEval")
    )

    imputed = impute_missing_benchmarks(candidate.benchmarks)

    # Determine whether primary or alternative benchmark is directly measured
    cand_bench_canon = {k.lower().replace("-", "_").replace(" ", "_"): v for k, v in candidate.benchmarks.items()}
    measured_list_canon = [m.lower().replace("-", "_").replace(" ", "_") for m in candidate.measured_benchmarks]

    if prim_key in measured_list_canon or prim_key in cand_bench_canon:
        selected_key = prim_key
        selected_name = prim_name
        is_measured = True
    elif alt_key in measured_list_canon or alt_key in cand_bench_canon:
        selected_key = alt_key
        selected_name = alt_name
        is_measured = True
    else:
        selected_key = prim_key
        selected_name = prim_name
        is_measured = False

    bench_score = float(imputed.get(selected_key, 50.0))

    ram_allocated = float(getattr(eval_result, "total_ram_mb", getattr(eval_result, "runtime_ram_mb", 0.0)))
    ram_headroom = float(budget_mb - ram_allocated)
    tps = float(eval_result.tokens_per_second)

    bullets: List[str] = []
    if candidate.proof_justification:
        bullets.append(candidate.proof_justification)
    else:
        bullets.append(f"Excels at {task_clean.replace('_', ' ')} workloads scoring {bench_score:.1f}% on {selected_name}.")

    bullets.append(
        f"Fits within hardware budget: {ram_allocated:.1f} MB allocated with {ram_headroom:.1f} MB safety headroom under {budget_mb:.1f} MB ceiling."
    )
    bullets.append(
        f"Projected generation throughput of {tps:.1f} tokens/sec on target backend."
    )

    return ProofCitation(
        model_id=candidate.repo_id,
        task_benchmark_name=selected_name,
        task_benchmark_score=round(bench_score, 1),
        is_measured=is_measured,
        ram_allocated_mb=round(ram_allocated, 1),
        ram_headroom_mb=round(ram_headroom, 1),
        tokens_per_second=round(tps, 1),
        justification_bullets=bullets,
    )


def generate_proof_citation_card(citation: ProofCitation) -> str:
    """Format a 3-pillar proof citation markdown card."""
    provenance = "[Measured]" if citation.is_measured else "[Estimated]"
    lines = [
        "#### 🏛️ 3-Pillar Proof Citation",
        f"- **Pillar 1 (Task Benchmark Proof):** {citation.task_benchmark_name} = `{citation.task_benchmark_score:.1f}` {provenance}",
        f"- **Pillar 2 (Hardware Headroom & Tok/s):** `{citation.ram_allocated_mb:.1f} MB` allocated, `{citation.ram_headroom_mb:.1f} MB` headroom margin, `{citation.tokens_per_second:.1f} tok/s`",
        "- **Pillar 3 (Why This Model Justification):**",
    ]
    for bullet in citation.justification_bullets:
        lines.append(f"  - {bullet}")
    return "\n".join(lines)


def generate_citation_card(
    candidate: CandidateModel,
    evaluation: Optional[SizingEvaluation] = None,
) -> str:
    """Generate a structured Markdown citation card for a candidate model."""
    lines: List[str] = [
        f"### 📦 `{candidate.repo_id}`",
        f"- **File:** `{candidate.filename}` ({candidate.size_mb:.1f} MB)",
        f"- **Quantization:** `{candidate.quant_method}`",
        f"- **Architecture:** `{candidate.arch_type}` ({candidate.param_count_billions:.2f}B params, active weight ratio: {candidate.active_weight_ratio:.2f})",
        f"- **Context Window:** {candidate.context_window:,} tokens",
        f"- **License:** `{candidate.license}`",
    ]

    # Academic citation & paper links
    if candidate.citation_doi:
        doi_url = f"https://doi.org/{candidate.citation_doi}"
        lines.append(f"- **Academic DOI:** [{candidate.citation_doi}]({doi_url})")
    if candidate.paper_url:
        lines.append(f"- **Research Paper:** [{candidate.paper_url}]({candidate.paper_url})")

    # Benchmarks
    if candidate.benchmarks:
        bench_lines = []
        for key, val in sorted(candidate.benchmarks.items()):
            label = key.replace("_", " ").title()
            if "elo" in key.lower():
                bench_lines.append(f"    - **{label}:** {val:.0f}")
            else:
                bench_lines.append(f"    - **{label}:** {val:.1f}")
        lines.append("- **Verified Benchmarks:**")
        lines.extend(bench_lines)

    # Hardware sizing breakdown if evaluation provided
    if evaluation is not None:
        status_str = "✅ Fits Budget" if evaluation.fits else "❌ Exceeds Budget"
        lines.extend([
            "- **Hardware Sizing Evaluation:**",
            f"    - **Status:** {status_str}",
            f"    - **Weights:** {evaluation.weights_mb:.1f} MB",
            f"    - **KV Cache:** {evaluation.kv_cache_mb:.1f} MB",
            f"    - **Overhead:** {evaluation.overhead_mb:.1f} MB",
            f"    - **Total Runtime Memory:** {evaluation.total_ram_mb:.1f} MB",
            f"    - **Budget Ceiling:** {evaluation.budget_ceiling_mb:.1f} MB",
            f"    - **Projected Throughput:** {evaluation.tokens_per_second:.1f} tokens/sec",
            f"    - **RAM Penalty:** {evaluation.ram_penalty:.2f}",
        ])
        if evaluation.rejection_reason:
            lines.append(f"    - **Rejection Reason:** {evaluation.rejection_reason}")

    return "\n".join(lines)


def generate_recommendations_markdown(
    scorecards: List[Scorecard],
    hardware: HardwareProfile,
    user_needs: UserNeeds,
) -> str:
    """Generate comprehensive RECOMMENDATIONS.md artifact content."""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines: List[str] = [
        "# Agent Companion Model Recommendations",
        f"\n*Generated by `companion-packager` on {now_str}*\n",
        "---",
        "\n## 1. Host Hardware Profile",
        f"- **Operating System:** {hardware.os_name}",
        f"- **CPU:** {hardware.cpu_brand} ({hardware.cpu_arch})",
        f"- **Cores:** {hardware.physical_cores} Physical / {hardware.logical_cores} Logical",
        f"- **CPU SIMD:** AVX-512: `{hardware.has_avx512}`, AVX2: `{hardware.has_avx2}`, NEON: `{hardware.has_neon}`",
        f"- **RAM:** {hardware.ram_avail_mb:.1f} MB Available / {hardware.ram_total_mb:.1f} MB Total",
        f"- **Disk Space Free:** {hardware.disk_free_gb:.1f} GB",
        f"- **Sustained Memory Bandwidth:** {hardware.bandwidth_gb_s:.1f} GB/s",
        f"- **Selected Backend Target:** `{hardware.backend_target}`",
    ]

    if hardware.gpus:
        lines.append("- **Detected GPUs:**")
        for gpu in hardware.gpus:
            lines.append(
                f"  - `[{gpu.index}]` {gpu.name} ({gpu.vendor.upper()}) - "
                f"{gpu.vram_total_mb:.1f} MB VRAM (Free: {gpu.vram_free_mb:.1f} MB), "
                f"Compute: {gpu.cuda_compute_capability or 'N/A'}, Driver: {gpu.driver_version or 'N/A'}"
            )
    else:
        lines.append("- **Detected GPUs:** None (Host will execute in CPU offloading mode)")

    lines.extend([
        "\n---",
        "\n## 2. User Workload & Operational Constraints",
        f"- **Task Domain:** `{user_needs.task}`",
        f"- **Optimization Priority:** `{user_needs.priority}`",
        f"- **Minimum Context Length:** {user_needs.min_context_tokens:,} tokens",
        f"- **Require Permissive License:** `{user_needs.require_permissive_license}`",
    ])
    if user_needs.custom_intent:
        lines.append(f"- **Custom Workload Intent:** `{user_needs.custom_intent}`")

    lines.extend([
        "\n---",
        "\n## 3. Ranked Model Recommendations",
    ])

    if not scorecards:
        lines.append("\n*No candidate models matched the host memory budget or user constraints.*")
    else:
        role_titles = {
            1: "🥇 Primary Recommended Model (Optimal utility score for host)",
            2: "🥈 High-Speed Alternative (Lighter footprint / higher throughput)",
            3: "🥉 Maximum-IQ Alternative (Deepest reasoning within hardware ceiling)",
        }

        for sc in scorecards:
            role_header = role_titles.get(sc.rank, f"Model Option (Rank #{sc.rank})")
            lines.extend([
                f"\n### {role_header}",
                f"**Final Utility Score:** `{sc.final_score:.3f}` | "
                f"**Task Fit:** `{sc.task_fit_score:.3f}` | "
                f"**Latency Score:** `{sc.latency_score:.3f}`\n",
                generate_citation_card(sc.candidate, sc.evaluation),
            ])
            proof_cit = sc.proof_citation
            if proof_cit is None:
                proof_cit = generate_proof_citation(
                    candidate=sc.candidate,
                    task=user_needs.task,
                    eval_result=sc.evaluation,
                    budget_mb=sc.evaluation.budget_ceiling_mb,
                )
            lines.append("\n" + generate_proof_citation_card(proof_cit))

    # Execution guide for primary recommendation
    if scorecards:
        primary = scorecards[0]
        lines.extend([
            "\n---",
            "\n## 4. Quickstart Execution Guide",
            "Launch the recommended model locally using `llama-server`:",
            "\n```bash",
            f"# 1. Download checkpoint",
            f"companion-packager download --model {primary.candidate.repo_id} --file {primary.candidate.filename}",
            f"\n# 2. Run local inference server",
            f"llama-server -m models/{primary.candidate.filename} -c {user_needs.min_context_tokens} --port 8000",
            "```",
        ])

    # Academic citations
    lines.extend([
        "\n---",
        "\n## 5. Academic References & Citations",
        "```bibtex",
        "@article{qwen2_5_coder_2024,",
        "  title={Qwen2.5-Coder Technical Report},",
        "  author={Qwen Team},",
        "  journal={arXiv preprint arXiv:2409.12186},",
        "  year={2024}",
        "}",
        "",
        "@article{deepseek_coder_v2_2024,",
        "  title={DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence},",
        "  author={Zhu, Qihao and others},",
        "  journal={arXiv preprint arXiv:2406.11931},",
        "  year={2024}",
        "}",
        "```\n",
    ])

    return "\n".join(lines)
