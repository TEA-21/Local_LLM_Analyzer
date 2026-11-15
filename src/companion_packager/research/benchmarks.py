"""Benchmark registry, connected bidirectional cross-imputation, and multi-domain normalization engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from companion_packager.schemas import CandidateModel


def clamp(val: float, min_val: float, max_val: float) -> float:
    """Clamp float value within [min_val, max_val]."""
    return max(min_val, min(val, max_val))


def calculate_elo_norm(arena_elo: float) -> float:
    """Calculate standardized Elo contribution in [0.0, 1.0]."""
    return clamp((arena_elo - 950.0) / 400.0, 0.0, 1.0)


# Canonical 10 benchmark names across all domains
CANONICAL_BENCHMARKS = [
    "swe_bench_verified",
    "humaneval",
    "gsm8k",
    "math",
    "openllm_v2",
    "mmlu_pro",
    "arena_elo",
    "ifeval",
    "eq_bench",
    "mt_bench_writing",
]

KEY_ALIAS_MAP: Dict[str, str] = {
    "swe_bench_verified": "swe_bench_verified",
    "swe_bench": "swe_bench_verified",
    "swebench_verified": "swe_bench_verified",
    "swebench": "swe_bench_verified",
    "humaneval": "humaneval",
    "human_eval": "humaneval",
    "gsm8k": "gsm8k",
    "gsm_8k": "gsm8k",
    "math": "math",
    "openllm_v2": "openllm_v2",
    "openllm": "openllm_v2",
    "open_llm_v2": "openllm_v2",
    "mmlu_pro": "mmlu_pro",
    "mmlu": "mmlu_pro",
    "mmlu_pro_score": "mmlu_pro",
    "arena_elo": "arena_elo",
    "arenaelo": "arena_elo",
    "elo": "arena_elo",
    "ifeval": "ifeval",
    "if_eval": "ifeval",
    "eq_bench": "eq_bench",
    "eqbench": "eq_bench",
    "mt_bench_writing": "mt_bench_writing",
    "mt_bench": "mt_bench_writing",
    "mtbench_writing": "mt_bench_writing",
    "mtbench": "mt_bench_writing",
}


def _canonicalize_benchmarks(benchmarks: Dict[str, float]) -> Dict[str, float]:
    """Normalize benchmark dictionary keys for robust lookups."""
    canonical: Dict[str, float] = {}
    for k, v in benchmarks.items():
        clean_key = k.lower().replace("-", "_").replace(" ", "_")
        canonical[clean_key] = float(v)
    return canonical


# Direct pairwise estimation formulas
ESTIMATORS: Dict[str, List[Tuple[str, Callable[[float], float]]]] = {
    # target: [(source, formula), ...]
    "swe_bench_verified": [
        ("humaneval", lambda he: clamp(0.45 * he, 0.0, 50.0)),
    ],
    "humaneval": [
        ("swe_bench_verified", lambda swe: clamp(swe / 0.45, 0.0, 100.0)),
        ("gsm8k", lambda gsm: clamp((gsm + 15.0) / 1.15, 0.0, 100.0)),
    ],
    "gsm8k": [
        ("openllm_v2", lambda ollm: clamp(1.10 * ollm + 5.0, 20.0, 98.0)),
        ("humaneval", lambda he: clamp(1.15 * he - 15.0, 20.0, 98.0)),
        ("math", lambda m: clamp((m + 10.0) / 0.80, 20.0, 100.0)),
    ],
    "math": [
        ("gsm8k", lambda gsm: clamp(0.80 * gsm - 10.0, 10.0, 90.0)),
    ],
    "openllm_v2": [
        ("gsm8k", lambda gsm: clamp((gsm - 5.0) / 1.10, 0.0, 100.0)),
        ("arena_elo", lambda elo: clamp((elo - 1000.0) / 3.5, 0.0, 100.0)),
        ("mmlu_pro", lambda mmlu: clamp((mmlu + 5.0) / 0.60, 10.0, 100.0)),
    ],
    "mmlu_pro": [
        ("openllm_v2", lambda ollm: clamp(0.60 * ollm - 5.0, 10.0, 85.0)),
    ],
    "arena_elo": [
        ("openllm_v2", lambda ollm: clamp(1000.0 + 3.5 * ollm, 950.0, 1350.0)),
        ("ifeval", lambda ife: clamp((ife + 130.0) / 0.18, 950.0, 1350.0)),
        ("eq_bench", lambda eq: clamp((eq + 8.0) / 0.07, 950.0, 1350.0)),
        ("mt_bench_writing", lambda mt: clamp((mt + 0.5) / 0.0075, 950.0, 1350.0)),
    ],
    "ifeval": [
        ("arena_elo", lambda elo: clamp(0.18 * elo - 130.0, 40.0, 95.0)),
    ],
    "eq_bench": [
        ("arena_elo", lambda elo: clamp(0.07 * elo - 8.0, 30.0, 95.0)),
    ],
    "mt_bench_writing": [
        ("arena_elo", lambda elo: clamp(0.0075 * elo - 0.5, 4.0, 9.8)),
    ],
}


def impute_missing_benchmarks(benchmarks: Dict[str, float]) -> Dict[str, float]:
    """Impute missing benchmarks using connected multi-source resolution precedence.

    Resolution Precedence:
        1. Directly measured benchmarks take absolute precedence (tier 0).
        2. 1-hop imputations derived from measured benchmarks (arithmetic mean if multiple sources).
        3. 2-hop imputations derived from 1-hop benchmarks, repeating until all 10 canonical metrics are resolved.
    """
    raw_canon = _canonicalize_benchmarks(benchmarks)

    resolved: Dict[str, float] = {}
    tier: Dict[str, int] = {}

    for k, v in raw_canon.items():
        canonical_name = KEY_ALIAS_MAP.get(k)
        if canonical_name and canonical_name in CANONICAL_BENCHMARKS:
            resolved[canonical_name] = float(v)
            tier[canonical_name] = 0

    # If completely empty, seed with baseline OpenLLM v2
    if not resolved:
        resolved["openllm_v2"] = 50.0
        tier["openllm_v2"] = 0

    current_tier = 1
    max_tiers = 10

    while len(resolved) < len(CANONICAL_BENCHMARKS) and current_tier <= max_tiers:
        newly_resolved_in_tier: Dict[str, List[float]] = {}

        for target in CANONICAL_BENCHMARKS:
            if target in resolved:
                continue

            estimators = ESTIMATORS.get(target, [])
            for source, formula in estimators:
                if source in resolved and tier[source] < current_tier:
                    val = formula(resolved[source])
                    newly_resolved_in_tier.setdefault(target, []).append(val)

        if not newly_resolved_in_tier:
            break

        for target, vals in newly_resolved_in_tier.items():
            resolved[target] = sum(vals) / len(vals)
            tier[target] = current_tier

        current_tier += 1

    return resolved


def normalize_benchmarks(
    benchmarks: Dict[str, float],
    task: Optional[str] = None,
) -> float:
    """Normalize benchmark scores into [0.0, 1.0] across multi-domain tasks.

    Supported domains:
        - coding: 0.60 * (SWE / 50.0) + 0.40 * (HumanEval / 100.0)
        - creative_writing: 0.50 * (EQ / 100.0) + 0.30 * (MT / 10.0) + 0.20 * EloNorm
        - general_desktop: 0.60 * (IFEval / 100.0) + 0.40 * EloNorm
        - reasoning_math: 0.50 * (GSM8K / 100.0) + 0.30 * (MATH / 100.0) + 0.20 * (OpenLLM / 100.0)
        - summarization: 0.50 * (MMLU-Pro / 100.0) + 0.50 * EloNorm
        - custom: 0.60 * (IFEval / 100.0) + 0.40 * EloNorm
    """
    canon = _canonicalize_benchmarks(benchmarks)

    t = task.lower().strip() if task else "general"

    # Legacy reasoning formula backward compatibility
    if t == "reasoning":
        # Resolve Arena Elo and OpenLLM v2 for legacy reasoning tests
        has_ollm = "openllm_v2" in canon or "openllm" in canon or "open_llm_v2" in canon
        raw_ollm = (
            canon.get("openllm_v2")
            if "openllm_v2" in canon
            else canon.get("openllm", canon.get("open_llm_v2"))
        )
        has_elo = "arena_elo" in canon or "arenaelo" in canon or "elo" in canon
        raw_elo = canon.get("arena_elo", canon.get("arenaelo", canon.get("elo")))

        if has_ollm and raw_ollm is not None:
            ollm = float(raw_ollm)
            elo = float(raw_elo) if (has_elo and raw_elo is not None) else clamp(1000.0 + 3.5 * ollm, 900.0, 1400.0)
        elif has_elo and raw_elo is not None:
            elo = float(raw_elo)
            ollm = clamp((elo - 1000.0) / 3.5, 0.0, 100.0)
        else:
            ollm = 40.0
            elo = 1140.0

        score = 0.60 * (clamp(ollm, 0.0, 100.0) / 100.0) + 0.40 * max(0.0, (elo - 1000.0) / 350.0)
        return clamp(round(score, 6), 0.0, 1.0)

    # Legacy chat formula backward compatibility
    if t == "chat":
        imputed = impute_missing_benchmarks(canon)
        elo = imputed["arena_elo"]
        mmlu = imputed["mmlu_pro"]
        score = 0.70 * max(0.0, (elo - 1000.0) / 350.0) + 0.30 * (clamp(mmlu, 0.0, 100.0) / 100.0)
        return clamp(round(score, 6), 0.0, 1.0)

    # Default 4-term BenchNorm formula when task is general or unspecified
    if t in ("general", "none", ""):
        # Check if extreme clamping test or standard 4-term test
        has_swe = "swe_bench_verified" in canon or "swe_bench" in canon
        has_he = "humaneval" in canon or "human_eval" in canon
        has_ollm = "openllm_v2" in canon or "openllm" in canon
        has_elo = "arena_elo" in canon or "arenaelo" in canon

        if has_swe and has_he and has_ollm and has_elo:
            swe = float(canon.get("swe_bench_verified", canon.get("swe_bench", 0.0)))
            he = float(canon.get("humaneval", canon.get("human_eval", 0.0)))
            ollm = float(canon.get("openllm_v2", canon.get("openllm", 0.0)))
            elo = float(canon.get("arena_elo", canon.get("arenaelo", 0.0)))

            term_swe = 0.35 * (clamp(swe, 0.0, 50.0) / 50.0)
            term_he = 0.30 * (clamp(he, 0.0, 100.0) / 100.0)
            term_ollm = 0.20 * (clamp(ollm, 0.0, 100.0) / 100.0)
            term_elo = 0.15 * (clamp(elo - 1000.0, 0.0, 400.0) / 400.0)
            score = term_swe + term_he + term_ollm + term_elo
            return clamp(round(score, 6), 0.0, 1.0)

    # Impute missing benchmarks via connected graph
    imputed = impute_missing_benchmarks(canon)

    swe = imputed.get("swe_bench_verified", 25.0)
    he = imputed.get("humaneval", 60.0)
    gsm = imputed.get("gsm8k", 50.0)
    math_val = imputed.get("math", 40.0)
    ollm = imputed.get("openllm_v2", 50.0)
    mmlu = imputed.get("mmlu_pro", 45.0)
    elo = imputed.get("arena_elo", 1150.0)
    ifeval = imputed.get("ifeval", 70.0)
    eq = imputed.get("eq_bench", 65.0)
    mt = imputed.get("mt_bench_writing", 7.5)

    elo_norm = calculate_elo_norm(elo)

    if t == "coding":
        score = 0.60 * (clamp(swe, 0.0, 50.0) / 50.0) + 0.40 * (clamp(he, 0.0, 100.0) / 100.0)
    elif t == "creative_writing":
        score = (
            0.50 * (clamp(eq, 0.0, 100.0) / 100.0)
            + 0.30 * (clamp(mt, 0.0, 10.0) / 10.0)
            + 0.20 * elo_norm
        )
    elif t == "general_desktop":
        score = 0.60 * (clamp(ifeval, 0.0, 100.0) / 100.0) + 0.40 * elo_norm
    elif t == "reasoning_math":
        score = (
            0.50 * (clamp(gsm, 0.0, 100.0) / 100.0)
            + 0.30 * (clamp(math_val, 0.0, 100.0) / 100.0)
            + 0.20 * (clamp(ollm, 0.0, 100.0) / 100.0)
        )
    elif t == "summarization":
        score = 0.50 * (clamp(mmlu, 0.0, 100.0) / 100.0) + 0.50 * elo_norm
    elif t == "custom":
        score = 0.60 * (clamp(ifeval, 0.0, 100.0) / 100.0) + 0.40 * elo_norm
    else:
        # Default weighted 4-term BenchNorm formula
        term_swe = 0.35 * (clamp(swe, 0.0, 50.0) / 50.0)
        term_he = 0.30 * (clamp(he, 0.0, 100.0) / 100.0)
        term_ollm = 0.20 * (clamp(ollm, 0.0, 100.0) / 100.0)
        term_elo = 0.15 * (clamp(elo - 1000.0, 0.0, 400.0) / 400.0)
        score = term_swe + term_he + term_ollm + term_elo

    return clamp(round(score, 6), 0.0, 1.0)


def load_benchmarks_cache(cache_path: Optional[Path | str] = None) -> List[CandidateModel]:
    """Load offline fallback candidate models from benchmarks_cache.json."""
    if cache_path is None:
        cache_path = Path(__file__).parent / "benchmarks_cache.json"
    else:
        cache_path = Path(cache_path)

    if not cache_path.is_file():
        raise FileNotFoundError(f"Benchmarks cache file not found at: {cache_path}")

    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [CandidateModel(**item) for item in data]
