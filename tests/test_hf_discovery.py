"""Unit and integration tests for Hugging Face dynamic model discovery and harvester."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest
from typer.testing import CliRunner

from companion_packager.cli import app
from companion_packager.research.hf_client import (
    detect_quant_and_file,
    discover_trending_models,
    estimate_model_layers,
    extract_arxiv_and_doi,
    extract_license,
    get_model_candidates,
    parse_param_count_from_name,
)
from companion_packager.schemas import CandidateModel, HardwareProfile

runner = CliRunner()


@pytest.fixture
def mock_hardware():
    return HardwareProfile(
        os_name="Windows",
        cpu_brand="Intel(R) Core(TM) i7-1165G7",
        cpu_arch="AMD64",
        physical_cores=4,
        logical_cores=8,
        has_avx512=True,
        has_avx2=True,
        has_neon=False,
        ram_total_mb=16384.0,
        ram_avail_mb=8192.0,
        disk_free_gb=50.0,
        bandwidth_gb_s=30.0,
        backend_target="cpu",
    )


class MockSibling:
    def __init__(self, rfilename, size=None, lfs_size=None):
        self.rfilename = rfilename
        self.size = size
        self.lfs = SimpleNamespace(size=lfs_size) if lfs_size is not None else None


class MockModelInfo:
    def __init__(self, model_id, siblings=None, tags=None):
        self.id = model_id
        self.modelId = model_id
        self.siblings = siblings or []
        self.tags = tags or []


def test_parse_param_count_from_name():
    """Verify parameter count extraction from varied model naming patterns."""
    assert parse_param_count_from_name("7B") == 7.0
    assert parse_param_count_from_name("14B") == 14.0
    assert parse_param_count_from_name("32B") == 32.0
    assert parse_param_count_from_name("1.5B") == 1.5
    assert parse_param_count_from_name("1.5b") == 1.5
    assert parse_param_count_from_name("3B") == 3.0
    assert parse_param_count_from_name("8B") == 8.0
    assert parse_param_count_from_name("0.5B") == 0.5
    assert parse_param_count_from_name("2.7B") == 2.7
    assert parse_param_count_from_name("70B") == 70.0
    assert parse_param_count_from_name("Qwen/Qwen2.5-Coder-7B-Instruct-GGUF") == 7.0
    assert parse_param_count_from_name("Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf") == 14.0
    assert parse_param_count_from_name("penkia/TernaryQuench-Qwen3.8-27B-GGUF") == 27.0
    assert parse_param_count_from_name("UnknownModelPattern") == 7.0
    assert parse_param_count_from_name("") == 7.0


def test_detect_quant_and_file_priorities():
    """Verify quantization format prioritization and size extraction."""
    # Q4_K_M preferred over Q8_0 and Q5_K_M
    s1 = MockSibling("model-Q8_0.gguf", size=8589934592)
    s2 = MockSibling("model-Q4_K_M.gguf", lfs_size=4294967296)  # 4096.0 MB
    s3 = MockSibling("model-Q5_K_M.gguf", size=5368709120)
    siblings = [s1, s2, s3]

    quant, fname, size_mb = detect_quant_and_file(siblings)
    assert quant == "Q4_K_M"
    assert fname == "model-Q4_K_M.gguf"
    assert pytest.approx(size_mb, abs=0.1) == 4096.0

    # Q5_K_M chosen when Q4_K_M is not present
    quant2, fname2, size_mb2 = detect_quant_and_file([s1, s3])
    assert quant2 == "Q5_K_M"
    assert fname2 == "model-Q5_K_M.gguf"
    assert pytest.approx(size_mb2, abs=0.1) == 5120.0

    # Fallback to first .gguf when non-standard quant
    fallback_sibling = MockSibling("model-IQ3_M.gguf", size=3145728000)
    quant3, fname3, size_mb3 = detect_quant_and_file([fallback_sibling])
    assert quant3 == "IQ3_M"
    assert fname3 == "model-IQ3_M.gguf"
    assert pytest.approx(size_mb3, abs=0.1) == 3000.0

    # Empty siblings list
    assert detect_quant_and_file([]) == ("Q4_K_M", "", 0.0)

    # Dict siblings compatibility
    dict_siblings = [
        {"rfilename": "test-Q4_0.gguf", "size": 3145728000},
        {"rfilename": "test-Q4_K_S.gguf", "lfs": {"size": 3670016000}},
    ]
    quant4, fname4, size_mb4 = detect_quant_and_file(dict_siblings)
    assert quant4 == "Q4_K_S"
    assert fname4 == "test-Q4_K_S.gguf"


def test_extract_arxiv_and_doi():
    """Verify extraction of arXiv tags and formatting into DOI and URL."""
    tags = ["gguf", "arxiv:2409.12188", "license:apache-2.0"]
    doi, url = extract_arxiv_and_doi(tags)
    assert doi == "10.48550/arXiv.2409.12188"
    assert url == "https://arxiv.org/abs/2409.12188"

    tags_v2 = ["arxiv:2401.02385v1"]
    doi2, url2 = extract_arxiv_and_doi(tags_v2)
    assert doi2 == "10.48550/arXiv.2401.02385v1"
    assert url2 == "https://arxiv.org/abs/2401.02385v1"

    no_arxiv = ["gguf", "llama", "license:mit"]
    assert extract_arxiv_and_doi(no_arxiv) == ("", "")
    assert extract_arxiv_and_doi([]) == ("", "")


def test_extract_license():
    """Verify license tag extraction and fallback."""
    assert extract_license(["license:apache-2.0", "gguf"]) == "apache-2.0"
    assert extract_license(["license:mit"]) == "mit"
    assert extract_license(["license:llama3.1"]) == "llama3.1"
    assert extract_license(["no-license"]) == "apache-2.0"
    assert extract_license([]) == "apache-2.0"


def test_estimate_model_layers():
    """Verify layer and attention head estimates across parameter tiers."""
    assert estimate_model_layers(1.5) == (28, 2, 128)
    assert estimate_model_layers(3.0) == (36, 2, 128)
    assert estimate_model_layers(7.0) == (28, 4, 128)
    assert estimate_model_layers(14.0) == (48, 8, 128)
    assert estimate_model_layers(32.0) == (64, 8, 128)
    assert estimate_model_layers(70.0) == (64, 8, 128)


def test_discover_trending_models_mocked():
    """Verify discover_trending_models parses Hub models into CandidateModel instances."""
    mock_model_1 = MockModelInfo(
        model_id="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        siblings=[
            MockSibling("qwen2.5-coder-7b-instruct-q4_k_m.gguf", lfs_size=4907335680),
            MockSibling("qwen2.5-coder-7b-instruct-q8_0.gguf", lfs_size=8157921280),
        ],
        tags=["gguf", "arxiv:2409.12188", "license:apache-2.0"],
    )

    mock_model_moe = MockModelInfo(
        model_id="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
        siblings=[
            MockSibling("DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf", lfs_size=10066329600),
        ],
        tags=["gguf", "arxiv:2406.11931", "license:deepseek-license"],
    )

    with patch("huggingface_hub.HfApi") as mock_api_cls:
        mock_api = MagicMock()
        mock_api.list_models.return_value = [mock_model_1, mock_model_moe]
        mock_api_cls.return_value = mock_api

        candidates = discover_trending_models(task="coding", limit=5)
        assert len(candidates) == 2

        c1 = candidates[0]
        assert c1.repo_id == "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"
        assert c1.filename == "qwen2.5-coder-7b-instruct-q4_k_m.gguf"
        assert c1.quant_method == "Q4_K_M"
        assert c1.param_count_billions == 7.0
        assert c1.active_weight_ratio == 1.0
        assert c1.citation_doi == "10.48550/arXiv.2409.12188"
        assert c1.paper_url == "https://arxiv.org/abs/2409.12188"
        assert c1.license == "apache-2.0"
        assert "swe_bench_verified" in c1.benchmarks

        c_moe = candidates[1]
        assert c_moe.active_weight_ratio == 0.15
        assert c_moe.arch_type == "deepseek_v2"
        assert c_moe.license == "deepseek-license"


def test_discover_trending_models_network_error_graceful():
    """Verify network or auth errors return empty list without crashing."""
    with patch("huggingface_hub.HfApi") as mock_api_cls:
        mock_api = MagicMock()
        mock_api.list_models.side_effect = RuntimeError("Connection timed out")
        mock_api_cls.return_value = mock_api

        candidates = discover_trending_models(task="coding")
        assert candidates == []


def test_get_model_candidates_deduplication():
    """Verify live search deduplicates against cached models by repo_id and filename."""
    # Candidate identical to one in cache
    duplicate_model = CandidateModel(
        repo_id="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        filename="qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        quant_method="Q4_K_M",
        param_count_billions=7.61,
        size_mb=4680.0,
    )
    # Distinct new model
    new_model = CandidateModel(
        repo_id="Qwen/QwQ-32B-Preview-GGUF",
        filename="qwq-32b-preview-q4_k_m.gguf",
        quant_method="Q4_K_M",
        param_count_billions=32.0,
        size_mb=19800.0,
    )

    with patch("companion_packager.research.hf_client.discover_trending_models") as mock_discover:
        mock_discover.return_value = [duplicate_model, new_model]

        candidates = get_model_candidates(
            offline=False,
            live_search=True,
            search_query="qwq",
        )

        # cached candidates + 1 new candidate (duplicate Qwen 7B filtered out)
        from companion_packager.research.benchmarks import load_benchmarks_cache
        assert len(candidates) == len(load_benchmarks_cache()) + 1
        repos = [c.repo_id for c in candidates]
        assert "Qwen/QwQ-32B-Preview-GGUF" in repos
        assert repos.count("Qwen/Qwen2.5-Coder-7B-Instruct-GGUF") == 1


def test_cli_analyze_search(mock_hardware):
    """Verify CLI analyze accepts --search and --headless in offline mode."""
    with patch("companion_packager.cli.detect_hardware", return_value=mock_hardware):
        result = runner.invoke(
            app,
            [
                "analyze",
                "--headless",
                "--offline",
                "--search",
                "qwq",
                "--limit",
                "5",
            ],
        )
        assert result.exit_code == 0
        assert "Candidate Model Sizing" in result.output
        assert "Top Recommended Model" in result.output
