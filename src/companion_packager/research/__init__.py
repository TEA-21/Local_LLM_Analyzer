"""Model research, Hugging Face Hub client, benchmarks, and citation synthesizer."""

from companion_packager.research.benchmarks import (
    calculate_elo_norm,
    impute_missing_benchmarks,
    load_benchmarks_cache,
    normalize_benchmarks,
)
from companion_packager.research.citations import (
    generate_citation_card,
    generate_proof_citation,
    generate_proof_citation_card,
    generate_recommendations_markdown,
)
from companion_packager.research.hf_client import (
    detect_quant_and_file,
    discover_trending_models,
    estimate_model_layers,
    extract_arxiv_and_doi,
    extract_license,
    get_model_candidates,
    parse_param_count_from_name,
)

__all__ = [
    "load_benchmarks_cache",
    "normalize_benchmarks",
    "impute_missing_benchmarks",
    "calculate_elo_norm",
    "get_model_candidates",
    "discover_trending_models",
    "parse_param_count_from_name",
    "detect_quant_and_file",
    "extract_arxiv_and_doi",
    "extract_license",
    "estimate_model_layers",
    "generate_citation_card",
    "generate_proof_citation",
    "generate_proof_citation_card",
    "generate_recommendations_markdown",
]
