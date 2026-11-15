"""Hugging Face Hub API client with offline airgapped fallback."""

import logging
import os
from typing import List, Optional

from companion_packager.research.benchmarks import load_benchmarks_cache
from companion_packager.schemas import CandidateModel

logger = logging.getLogger(__name__)


def get_model_candidates(
    offline: bool = False,
    hf_token: Optional[str] = None,
) -> List[CandidateModel]:
    """Retrieve candidate models from Hugging Face Hub or offline fallback cache.

    If offline=True or network is unreachable or authentication fails (401/403 for gated models),
    gracefully falls back to pre-packaged benchmarks_cache.json.
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

    # Online refinement: attempt to verify file sizes against LFS metadata
    updated_candidates: List[CandidateModel] = []

    for candidate in cached_candidates:
        cand_copy = candidate.model_copy(deep=True)
        try:
            info = api.model_info(cand_copy.repo_id, files_metadata=True)
            if info and info.siblings:
                for sibling in info.siblings:
                    rfilename = getattr(sibling, "rfilename", "")
                    if rfilename == cand_copy.filename or rfilename.endswith(cand_copy.filename):
                        # Extract exact LFS size
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

    return updated_candidates
