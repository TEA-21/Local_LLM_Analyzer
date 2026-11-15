"""Hugging Face Hub API client with offline airgapped fallback and dynamic model discovery."""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from companion_packager.interview.questionnaire import parse_custom_intent
from companion_packager.research.benchmarks import load_benchmarks_cache
from companion_packager.schemas import CandidateModel
from companion_packager.sizing.formulas import calculate_weights_mb

logger = logging.getLogger(__name__)

QUANT_PRIORITY: List[str] = ["Q4_K_M", "Q5_K_M", "Q4_K_S", "Q4_0", "Q8_0", "Q2_K"]

TASK_QUERY_BUNDLES: Dict[str, List[str]] = {
    "coding": ["coder", "code", "dev"],
    "creative_writing": ["story", "creative", "writing", "prose", "rp"],
    "general_desktop": ["instruct", "assistant", "general"],
    "reasoning_math": ["reasoning", "math", "logic", "distill"],
    "summarization": ["summary", "summarization", "long-context"],
    "reasoning": ["reasoning", "math", "logic", "distill"],
    "chat": ["instruct", "assistant", "general"],
}

TASK_SEARCH_QUERIES: Dict[str, str] = {
    "coding": "coder",
    "creative_writing": "story",
    "general_desktop": "instruct",
    "reasoning_math": "reasoning",
    "summarization": "summary",
    "reasoning": "reasoning",
    "chat": "instruct",
}


def quant_to_bits_per_weight(quant_method: str) -> float:
    """Map quantization method identifier to approximate bits per weight."""
    quant_upper = quant_method.upper()
    if "Q2" in quant_upper:
        return 2.6
    elif "Q3" in quant_upper or "IQ3" in quant_upper:
        return 3.3
    elif "Q5" in quant_upper:
        return 5.5
    elif "Q8" in quant_upper:
        return 8.0
    elif "Q6" in quant_upper:
        return 6.5
    return 4.5


def parse_param_count_from_name(name_or_repo: str) -> float:
    """Extract parameter count in billions from model name or repository string.

    Matches patterns like '7B', '14B', '32B', '1.5B', '3B', '8B', '0.5B', '2.7B', '70B'.
    Defaults to 7.0 if unparseable.
    """
    if not name_or_repo:
        return 7.0

    # Look for numbers immediately followed by b/B, bounded by start/delimiters
    matches = re.findall(
        r"(?:^|[\s_\-/])(\d+(?:\.\d+)?)\s*[bB](?:[\s_\-/\.]|$)",
        name_or_repo,
    )
    if matches:
        try:
            return float(matches[-1])
        except (ValueError, TypeError):
            pass

    return 7.0


def _extract_sibling_info(sibling: Any) -> Tuple[str, float]:
    """Defensively extract filename and size in MB from a sibling object or dict."""
    rfilename = getattr(sibling, "rfilename", None)
    if not rfilename and isinstance(sibling, dict):
        rfilename = sibling.get("rfilename") or sibling.get("path") or sibling.get("name")
    if not rfilename and hasattr(sibling, "name"):
        rfilename = getattr(sibling, "name", "")
    rfilename = str(rfilename or "")

    size_bytes = None
    lfs = getattr(sibling, "lfs", None)
    if lfs is None and isinstance(sibling, dict):
        lfs = sibling.get("lfs")

    if lfs:
        if isinstance(lfs, dict):
            size_bytes = lfs.get("size")
        else:
            size_bytes = getattr(lfs, "size", None)

    if size_bytes is None:
        if isinstance(sibling, dict):
            size_bytes = sibling.get("size")
        else:
            size_bytes = getattr(sibling, "size", None)

    size_mb = 0.0
    if size_bytes:
        try:
            size_mb = round(float(size_bytes) / (1024.0 * 1024.0), 1)
        except (ValueError, TypeError):
            size_mb = 0.0

    return rfilename, size_mb


def detect_quant_and_file(siblings: list) -> Tuple[str, str, float]:
    """Scan sibling files ending with .gguf and extract recommended quant and size.

    Prioritizes quant methods in order:
    Q4_K_M -> Q5_K_M -> Q4_K_S -> Q4_0 -> Q8_0 -> Q2_K -> first .gguf.
    Extracts filename and size in MB from sibling.lfs.size or sibling.size.
    Returns: (quant_method, filename, size_mb).
    """
    if not siblings:
        return ("Q4_K_M", "", 0.0)

    gguf_files: List[Tuple[str, float]] = []
    for s in siblings:
        fname, smb = _extract_sibling_info(s)
        if fname.lower().endswith(".gguf") and not fname.lower().startswith("mmproj"):
            gguf_files.append((fname, smb))

    # If all were mmproj or empty, check without mmproj exclusion
    if not gguf_files:
        for s in siblings:
            fname, smb = _extract_sibling_info(s)
            if fname.lower().endswith(".gguf"):
                gguf_files.append((fname, smb))

    if not gguf_files:
        return ("Q4_K_M", "", 0.0)

    # 1. Search by priority quant methods
    for target_quant in QUANT_PRIORITY:
        for fname, smb in gguf_files:
            if target_quant in fname.upper():
                return (target_quant, fname, smb)

    # 2. Fallback to first .gguf file
    first_fname, first_smb = gguf_files[0]
    m = re.search(r"(I?Q\d+_[A-Z0-9_]+|Q\d+_\d+)", first_fname.upper())
    detected_quant = m.group(1) if m else "Q4_K_M"
    return (detected_quant, first_fname, first_smb)


def extract_arxiv_and_doi(tags: list[str]) -> Tuple[str, str]:
    """Scan model tags for arxiv IDs and return (doi, paper_url).

    Returns (doi, paper_url) where doi = f"10.48550/arXiv.{arxiv_id}"
    and paper_url = f"https://arxiv.org/abs/{arxiv_id}".
    """
    if not tags:
        return ("", "")

    for tag in tags:
        if isinstance(tag, str):
            tag_clean = tag.strip()
            m = re.match(r"^arxiv:(\d{4}\.\d{4,5}(?:v\d+)?|\d{5,7})$", tag_clean, re.IGNORECASE)
            if m:
                arxiv_id = m.group(1)
                return (f"10.48550/arXiv.{arxiv_id}", f"https://arxiv.org/abs/{arxiv_id}")
            elif tag_clean.lower().startswith("arxiv:"):
                arxiv_id = tag_clean.split(":", 1)[1].strip()
                if arxiv_id:
                    return (f"10.48550/arXiv.{arxiv_id}", f"https://arxiv.org/abs/{arxiv_id}")

    return ("", "")


def extract_license(tags: list[str]) -> str:
    """Extract license string from tags matching license:<name>."""
    if not tags:
        return "apache-2.0"

    for tag in tags:
        if isinstance(tag, str) and tag.lower().startswith("license:"):
            parts = tag.split(":", 1)
            if len(parts) == 2 and parts[1].strip():
                return parts[1].strip().lower()

    return "apache-2.0"


def estimate_model_layers(
    param_billions: float,
    is_moe: bool = False,
) -> Tuple[int, int, int]:
    """Estimate architecture hyperparameters: (num_layers, num_kv_heads, head_dim)."""
    if param_billions < 2.0:
        return (28, 2, 128)
    elif param_billions < 5.0:
        return (36, 2, 128)
    elif param_billions < 10.0:
        return (28, 4, 128)
    elif param_billions < 20.0:
        return (48, 8, 128)
    else:
        return (64, 8, 128)


def impute_baseline_benchmarks(param_billions: float, task: str = "coding") -> Dict[str, float]:
    """Impute baseline benchmark scores based on parameter count and task."""
    if param_billions <= 2.0:
        base = {"swe_bench_verified": 18.5, "humaneval": 52.0, "openllm_v2": 44.0, "arena_elo": 1080.0}
    elif param_billions <= 4.0:
        base = {"swe_bench_verified": 26.0, "humaneval": 68.0, "openllm_v2": 54.0, "arena_elo": 1140.0}
    elif param_billions <= 9.0:
        base = {"swe_bench_verified": 39.0, "humaneval": 88.0, "openllm_v2": 67.0, "arena_elo": 1220.0}
    elif param_billions <= 16.0:
        base = {"swe_bench_verified": 46.0, "humaneval": 92.0, "openllm_v2": 74.0, "arena_elo": 1280.0}
    elif param_billions <= 35.0:
        base = {"swe_bench_verified": 50.0, "humaneval": 94.0, "openllm_v2": 80.0, "arena_elo": 1310.0}
    else:
        base = {"swe_bench_verified": 54.0, "humaneval": 95.0, "openllm_v2": 83.0, "arena_elo": 1340.0}

    return base


def discover_trending_models(
    task: str = "coding",
    search_query: Optional[str] = None,
    limit: int = 15,
    hf_token: Optional[str] = None,
    custom_intent: str = "",
) -> List[CandidateModel]:
    """Discover trending GGUF model checkpoints from Hugging Face Hub."""
    token = hf_token or os.getenv("HF_TOKEN")

    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.warning("huggingface_hub not installed. Dynamic discovery unavailable.")
        return []

    try:
        api = HfApi(token=token)
    except Exception as exc:
        logger.warning("Failed to initialize HfApi for discovery: %s", exc)
        return []

    if search_query:
        query = search_query
    elif task.lower().strip() == "custom":
        custom_terms = parse_custom_intent(custom_intent)
        query = custom_terms[0] if custom_terms else "instruct"
    else:
        bundle = TASK_QUERY_BUNDLES.get(task.lower().strip(), ["coder"])
        query = bundle[0]

    # Defensive call to api.list_models supporting diverse hub/mock configurations
    model_iter = None
    try:
        model_iter = api.list_models(
            filter="gguf",
            search=query,
            sort="trending_score",
            limit=limit,
            full=True,
        )
    except TypeError:
        try:
            model_iter = api.list_models(
                filter="gguf",
                search=query,
                sort="trending",
                direction=-1,
                limit=limit,
                full=True,
            )
        except Exception:
            try:
                model_iter = api.list_models(filter="gguf", search=query, limit=limit, full=True)
            except Exception as exc:
                logger.warning("Failed to list models with fallback: %s", exc)
                return []
    except Exception as exc:
        try:
            model_iter = api.list_models(
                filter="gguf",
                search=query,
                sort="trending",
                limit=limit,
                full=True,
            )
        except Exception:
            try:
                model_iter = api.list_models(filter="gguf", search=query, limit=limit, full=True)
            except Exception as sub_exc:
                logger.warning("Failed to list models from Hugging Face Hub (%s, %s)", exc, sub_exc)
                return []

    if model_iter is None:
        return []

    discovered: List[CandidateModel] = []

    try:
        for model in model_iter:
            repo_id = getattr(model, "id", None) or getattr(model, "modelId", "")
            if not repo_id:
                continue

            siblings = getattr(model, "siblings", None) or []
            quant_method, filename, size_mb = detect_quant_and_file(siblings)
            if not filename:
                continue

            param_count = parse_param_count_from_name(repo_id)
            if param_count == 7.0 and filename:
                parsed_from_file = parse_param_count_from_name(filename)
                if parsed_from_file != 7.0:
                    param_count = parsed_from_file

            tags = getattr(model, "tags", None) or []
            doi, paper_url = extract_arxiv_and_doi(tags)
            license_str = extract_license(tags)

            repo_lower = repo_id.lower()
            is_moe = any(k in repo_lower for k in ["v2-lite", "moe", "deepseek-v2", "deepseek_v2", "mixtral"])
            active_weight_ratio = 0.15 if is_moe else 1.0
            arch_type = "deepseek_v2" if ("deepseek" in repo_lower and is_moe) else ("moe" if is_moe else "dense")

            num_layers, num_kv_heads, head_dim = estimate_model_layers(param_count, is_moe=is_moe)

            # Estimate context window
            context_window = 32768
            if any(k in repo_lower for k in ["128k", "131k"]):
                context_window = 131072
            elif any(k in repo_lower for k in ["64k"]):
                context_window = 65536
            elif any(k in repo_lower for k in ["16k"]):
                context_window = 16384
            elif any(k in repo_lower for k in ["8k"]):
                context_window = 8192

            # Fallback size estimation if missing in metadata
            if size_mb <= 0.0:
                bits = quant_to_bits_per_weight(quant_method)
                size_mb = round(calculate_weights_mb(param_count, bits), 1)

            benchmarks = impute_baseline_benchmarks(param_count, task=task)

            candidate = CandidateModel(
                repo_id=repo_id,
                filename=filename,
                quant_method=quant_method,
                param_count_billions=param_count,
                active_weight_ratio=active_weight_ratio,
                size_mb=size_mb,
                context_window=context_window,
                arch_type=arch_type,
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                mla_v_head_dim=128,
                mla_kv_lora_rank=512,
                mla_rope_head_dim=64,
                benchmarks=benchmarks,
                citation_doi=doi,
                paper_url=paper_url,
                license=license_str,
            )
            discovered.append(candidate)
    except Exception as exc:
        logger.warning("Error iterating models from Hugging Face Hub: %s", exc)

    return discovered


def get_model_candidates(
    offline: bool = False,
    hf_token: Optional[str] = None,
    live_search: bool = False,
    search_query: Optional[str] = None,
    task: str = "coding",
    limit: int = 10,
    custom_intent: str = "",
) -> List[CandidateModel]:
    """Retrieve candidate models from Hugging Face Hub or offline fallback cache.

    If offline=True, strictly loads local airgapped benchmarks cache.
    If live_search=True or search_query is supplied, discovers live trending models
    and merges them with cached models, deduplicating by repository and filename.
    """
    cached_candidates = load_benchmarks_cache()

    if offline:
        logger.info("Operating in offline airgapped mode. Returning cached model candidates.")
        return cached_candidates

    token = hf_token or os.getenv("HF_TOKEN")

    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
    except ImportError:
        logger.warning("huggingface_hub not installed. Falling back to cached candidates.")
        return cached_candidates

    try:
        api = HfApi(token=token)
    except Exception as exc:
        logger.warning("Failed to initialize HfApi (%s). Falling back to cached candidates.", exc)
        return cached_candidates

    # Online refinement: attempt to verify file sizes against LFS metadata for cached candidates
    updated_candidates: List[CandidateModel] = []

    for candidate in cached_candidates:
        cand_copy = candidate.model_copy(deep=True)
        try:
            info = api.model_info(cand_copy.repo_id, files_metadata=True)
            if info and info.siblings:
                for sibling in info.siblings:
                    rfilename = getattr(sibling, "rfilename", "")
                    if rfilename == cand_copy.filename or rfilename.endswith(cand_copy.filename):
                        lfs = getattr(sibling, "lfs", None)
                        if lfs and hasattr(lfs, "size") and lfs.size:
                            cand_copy.size_mb = round(float(lfs.size) / (1024.0 * 1024.0), 1)
                        elif hasattr(sibling, "size") and sibling.size:
                            cand_copy.size_mb = round(float(sibling.size) / (1024.0 * 1024.0), 1)
                        break
        except (GatedRepoError, HfHubHTTPError) as http_err:
            logger.warning(
                "Access restricted or gated repository for %s (%s). Using cached metadata.",
                cand_copy.repo_id,
                http_err,
            )
        except RepositoryNotFoundError as not_found:
            logger.warning(
                "Repository %s not found on Hugging Face Hub (%s). Using cached metadata.",
                cand_copy.repo_id,
                not_found,
            )
        except Exception as exc:
            logger.warning(
                "Network/API error fetching %s (%s). Using cached metadata.",
                cand_copy.repo_id,
                exc,
            )

        updated_candidates.append(cand_copy)

    # If live discovery requested, fetch and merge deduplicated trending candidates
    if live_search or search_query:
        discovered = discover_trending_models(
            task=task,
            search_query=search_query,
            limit=limit,
            hf_token=hf_token,
            custom_intent=custom_intent,
        )

        existing_keys = {
            (c.repo_id.lower(), c.filename.lower()) for c in updated_candidates
        }
        existing_repos = {c.repo_id.lower() for c in updated_candidates}

        for model in discovered:
            key = (model.repo_id.lower(), model.filename.lower())
            if key not in existing_keys and model.repo_id.lower() not in existing_repos:
                updated_candidates.append(model)
                existing_keys.add(key)
                existing_repos.add(model.repo_id.lower())

    return updated_candidates
