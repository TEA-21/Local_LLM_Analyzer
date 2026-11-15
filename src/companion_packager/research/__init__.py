"""Model research, Hugging Face Hub client, benchmarks, and citation synthesizer."""

from companion_packager.research.benchmarks import (
    load_benchmarks_cache,
    normalize_benchmarks,
)
from companion_packager.research.citations import (
    generate_citation_card,
    generate_recommendations_markdown,
)
from companion_packager.research.hf_client import get_model_candidates

__all__ = [
    "load_benchmarks_cache",
    "normalize_benchmarks",
    "get_model_candidates",
    "generate_citation_card",
    "generate_recommendations_markdown",
]
