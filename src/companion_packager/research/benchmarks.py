"""Benchmark registry, two-way imputation, and normalization engine."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from companion_packager.schemas import CandidateModel


def clamp(val: float, min_val: float, max_val: float) -> float:
    """Clamp float value within [min_val, max_val]."""
    return max(min_val, min(val, max_val))


def _canonicalize_benchmarks(benchmarks: Dict[str, float]) -> Dict[str, float]:
    """Normalize benchmark dictionary keys for robust lookups."""
    canonical: Dict[str, float] = {}
    for k, v in benchmarks.items():
        clean_key = k.lower().replace("-", "_").replace(" ", "_")
        canonical[clean_key] = float(v)
    return canonical


def normalize_benchmarks(
    benchmarks: Dict[str, float],
    task: Optional[str] = None,
) -> float:
    """Normalize benchmark scores into [0.0, 1.0] with robust two-way imputation.

    Guarantees zero NoneType/KeyError failures by imputing missing metrics:
    - SWE-bench Verified <-> HumanEval
    - Arena Elo <-> OpenLLM v2
    - MMLU-Pro defaults to OpenLLM v2 or baseline
    """
    canon = _canonicalize_benchmarks(benchmarks)

    # 1. Resolve SWE-bench Verified and HumanEval
    has_swe = "swe_bench_verified" in canon or "swe_bench" in canon or "swebench" in canon
    raw_swe = (
        canon.get("swe_bench_verified")
        if "swe_bench_verified" in canon
        else canon.get("swe_bench", canon.get("swebench"))
    )

    has_he = "humaneval" in canon or "human_eval" in canon
    raw_he = canon.get("humaneval", canon.get("human_eval"))

    if has_swe and raw_swe is not None:
        swe = float(raw_swe)
        if has_he and raw_he is not None:
            he = float(raw_he)
        else:
            # Impute HumanEval from SWE-bench
            he = clamp(swe / 0.45, 0.0, 100.0)
    elif has_he and raw_he is not None:
        he = float(raw_he)
        # Impute SWE-bench from HumanEval
        swe = clamp(0.45 * he, 0.0, 50.0)
    else:
        # Default baseline if neither exists
        he = 60.0
        swe = clamp(0.45 * he, 0.0, 50.0)

    # 2. Resolve Arena Elo and OpenLLM v2
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
        if has_elo and raw_elo is not None:
            elo = float(raw_elo)
        else:
            # Impute Arena Elo from OpenLLM v2
            elo = clamp(1000.0 + 3.5 * ollm, 900.0, 1400.0)
    elif has_elo and raw_elo is not None:
        elo = float(raw_elo)
        # Impute OpenLLM v2 from Arena Elo
        ollm = clamp((elo - 1000.0) / 3.5, 0.0, 100.0)
    else:
        ollm = 40.0
        elo = 1000.0 + 3.5 * ollm  # 1140.0

    # 3. Resolve MMLU-Pro
    has_mmlu = "mmlu_pro" in canon or "mmlu" in canon or "mmlu_pro_score" in canon
    raw_mmlu = canon.get("mmlu_pro", canon.get("mmlu", canon.get("mmlu_pro_score")))
    if has_mmlu and raw_mmlu is not None:
        mmlu = float(raw_mmlu)
    else:
        mmlu = ollm if ollm > 0.0 else 45.0

    # 4. Compute BenchNorm
    t = task.lower() if task else "general"

    if t == "coding":
        score = 0.60 * (clamp(swe, 0.0, 50.0) / 50.0) + 0.40 * (clamp(he, 0.0, 100.0) / 100.0)
    elif t == "reasoning":
        score = 0.60 * (clamp(ollm, 0.0, 100.0) / 100.0) + 0.40 * max(0.0, (elo - 1000.0) / 350.0)
    elif t == "chat":
        score = 0.70 * max(0.0, (elo - 1000.0) / 350.0) + 0.30 * (clamp(mmlu, 0.0, 100.0) / 100.0)
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
