# Agent Companion Packager - Progress & Tracking

## Current Active Phase & Status
- **Current Phase:** Phase 8 Complete: Dynamic Hugging Face Discovery Engine & Live Leaderboard Harvester
- **Status:** PASSED. Dynamic Hugging Face Hub discovery subsystem implemented in `src/companion_packager/research/hf_client.py` and CLI commands (`analyze`, `recommend` with `--live`, `--no-live`, `--search`, `--limit`). Unit test suite in `tests/test_hf_discovery.py` (9 tests) and CLI integration tests in `tests/test_cli.py` (2 new tests) implemented and verified with 100% test pass rate (124/124 tests passing across full suite in 33.14s). Distributable wheel `companion_packager-0.1.0-py3-none-any.whl` compiled and verified.
- **argon_audit_runs:** 1

---

## Completed Steps & Exact Techniques/Libraries Used
1. **Packaging & Scaffolding (`pyproject.toml`):**
   - Configured `hatchling.build` backend, Python >= 3.11, entrypoint `companion-packager = "companion_packager.cli:app"`.
   - Declared core dependencies: `typer>=0.12.0`, `rich>=13.7.0`, `pydantic>=2.7.0`, `psutil>=5.9.8`, `py-cpuinfo>=9.0.0`, `huggingface-hub>=0.23.0`, `httpx>=0.27.0`, `build>=1.2.0`, `hatchling>=1.21.0`.
   - Verified wheel building with `python -m build --wheel` produces `companion_packager-0.1.0-py3-none-any.whl`.
2. **Pydantic v2 Domain Schemas (`src/companion_packager/schemas.py` & `models/schemas.py`):**
   - Implemented `GPUInfo`, `HardwareProfile`, `UserNeeds`, `CandidateModel`, `SizingEvaluation`, and `Scorecard`.
   - Dual-exported via `companion_packager.schemas` and `companion_packager.models.schemas`.
3. **Mathematical Sizing & Throughput Formulas (`src/companion_packager/sizing/formulas.py`):**
   - `calculate_weights_mb`: Exact parameter count and bits-per-weight equation.
   - `calculate_kv_cache_mb`: Standard GQA/MHA and DeepSeek MLA attention formulations with FP16 and Q8_0 precision fallback.
   - `calculate_budget_ceiling_mb`: GPU offloading formula `min(total * 0.80, free * 0.95)` and hard CPU memory invariant `min(avail_ram * 0.80, 6144.0 MB)`.
   - `estimate_tokens_per_second`: Memory bandwidth bound throughput with MoE `active_weight_ratio` scaling and neuromorphic sparsity speedup.
   - `calculate_ram_penalty`: 3-stage piecewise penalty (0.0 <= 87.5%, linear gradient to 100%, 1000.0 disqualification).
   - `evaluate_sizing`: Comprehensive feasibility and throughput evaluation returning `SizingEvaluation`.
4. **Zero-PyTorch GPU Detection Subsystem (`src/companion_packager/hardware/gpu_detector.py`):**
   - Completely decoupled from torch/CUDA runtimes.
   - Cross-platform NVIDIA detection via `nvidia-smi` CSV queries with robust timeout and numeric parsing.
   - macOS Apple Silicon unified memory detection via `sysctl -n hw.memsize` and `machdep.cpu.brand_string`.
   - Windows PowerShell CIM (`Win32_VideoController`) paired with defensive 64-bit registry query for `HardwareInformation.qwMemorySize` (supporting both integer `REG_DWORD`/`REG_QWORD` and 8-byte little-endian `REG_BINARY` types).
   - Linux DRM sysfs probing via `/sys/class/drm/card*/device/mem_info_vram_total`.
5. **Hardware Profiling & Heuristics (`src/companion_packager/hardware/detector.py` & `__init__.py`):**
   - `detect_hardware()`: Integrates `psutil` virtual memory, CPU core counts, `cpuinfo` SIMD flags (`has_avx512`, `has_avx2`, `has_neon`), zero-PyTorch GPU probing, and root/target disk space.
   - Backend target selector: Deterministically selects `cuda` (NVIDIA dGPU >= 4GB), `metal` (Apple Silicon), `rocm` (AMD dGPU >= 4GB), or `cpu` fallback.
   - `get_memory_bandwidth_gb_s()`: Deterministic sustained bandwidth mapping across GPUs, Apple Silicon, and CPU architectures.
   - `check_disk_space_preflight()`: Strict ratio validation for build (2.5x) vs download (1.5x).
6. **Airgapped Benchmarks Cache (`src/companion_packager/research/benchmarks_cache.json`):**
   - Bundles 7 verified candidate models stratified across hardware envelopes:
     - Sub-6GB CPU: Qwen2.5-Coder 7B (4680 MB), 3B (2050 MB), 1.5B (1150 MB).
     - Mid-Tier GPU: DeepSeek-Coder-V2-Lite MoE (9600 MB, active ratio 0.15, MLA architecture), Qwen2.5-Coder 14B (9200 MB).
     - High-Tier GPU: TernaryQuench-Qwen3.8-27B Q2_K (10360 MB), Qwen2.5-Coder 32B (19800 MB).
   - Packaged directly inside wheel distribution for 100% airgapped runtime.
7. **Two-Way Benchmark Imputation & BenchNorm Engine (`src/companion_packager/research/benchmarks.py`):**
   - Two-way imputation prevents KeyError/NoneType:
     - `SWE-bench_Verified` <-> `HumanEval` (`swe = clamp(0.45 * he, 0, 50)` / `he = clamp(swe / 0.45, 0, 100)`).
     - `Arena_Elo` <-> `OpenLLM_v2` (`elo = clamp(1000 + 3.5 * ollm, 900, 1400)` / `ollm = clamp((elo - 1000)/3.5, 0, 100)`).
     - `MMLU-Pro` fallback to `OpenLLM_v2` or baseline.
   - Normalization into `[0.0, 1.0]` across general 4-term weighted default and task-specific formulas (coding, reasoning, chat).
8. **Dynamic Hugging Face Hub Discovery Engine & Harvester (`src/companion_packager/research/hf_client.py`):**
   - `parse_param_count_from_name`: Robust regex extraction of parameter scales in billions ("7B", "14B", "32B", "1.5b", "70B"), defaulting safely to 7.0 if unparseable.
   - `detect_quant_and_file`: Scans sibling files ending with `.gguf` (excluding multimodal `mmproj` files), prioritizing quantization formats (`Q4_K_M` -> `Q5_K_M` -> `Q4_K_S` -> `Q4_0` -> `Q8_0` -> `Q2_K` -> first `.gguf`), extracting LFS file size or direct size in MB.
   - `extract_arxiv_and_doi`: Scans repository tags for `arxiv:XXXX.YYYYY` or `arxiv:XXXXX`, generating canonical DOI (`10.48550/arXiv.{id}`) and paper URLs (`https://arxiv.org/abs/{id}`).
   - `extract_license`: Parses tags for `license:<name>`, defaulting cleanly to `apache-2.0`.
   - `estimate_model_layers`: Rule-based architecture mapping for layers, KV heads, and head dimensions across parameter envelopes (<2B -> (28, 2, 128), <5B -> (36, 2, 128), <10B -> (28, 4, 128), <20B -> (48, 8, 128), else -> (64, 8, 128)).
   - `quant_to_bits_per_weight`: Accurately maps quantization methods to bits-per-weight for mathematical size estimation when remote LFS metadata is unpopulated.
   - `discover_trending_models`: Queries Hugging Face Hub with curated task terms (`coder`, `reasoning`, `instruct`) or custom search query, handles API pagination/sorting defensively (`trending_score` with fallback to `trending` and un-sorted), detects MoE architectures (`v2-lite`, `moe`, `deepseek-v2` setting active ratio to 0.15), imputes baseline benchmark metrics, and returns list of `CandidateModel`.
   - `get_model_candidates`: Supports `live_search` and `search_query`, deduplicates discovered candidates against cached models by repo ID and filename, while preserving airgapped offline guarantees.
9. **Citation Cards & Markdown Report Synthesizer (`src/companion_packager/research/citations.py`):**
   - `generate_citation_card()`: Formats model details, architecture parameters, academic DOI, paper URLs, verified benchmarks, and hardware sizing breakdown.
   - `generate_recommendations_markdown()`: Assembles complete `RECOMMENDATIONS.md` summarizing scanned hardware, workload constraints, top 3 ranked models, quickstart execution instructions (`llama-server`), and BibTeX academic citations.
10. **Interactive & Headless User Needs Discovery (`src/companion_packager/interview/questionnaire.py`):**
    - `ask_user_needs()`: Exposes dual-mode operation:
      - Interactive mode: Rich `Prompt.ask` for task domain (`coding`, `reasoning`, `chat`), optimization goal (`quality`, `balanced`, `speed`), context length (`16384`, `32768`, `8192`, `4096`), and `Confirm.ask` for permissive license constraint.
      - Headless / automated mode: Uses defaults or CLI arguments without blocking.
    - Full input validation rejecting unknown task types, priorities, or non-positive context limits.
11. **Multi-Backend Sizing Matrix & Recommender Engine (`src/companion_packager/sizing/recommender.py`):**
    - `rank_candidates()`:
      - Case-insensitive filtering against `PERMISSIVE_LICENSES` (`{"apache-2.0", "mit", "bsd-3-clause", "bsd-2-clause"}`).
      - Context window threshold enforcement (disqualifying models whose context window is smaller than requested context).
      - Domain task fit via `normalize_benchmarks` and latency scoring `min(tok_s / 50.0, 1.0)`.
      - Priority weighting:
        - `quality`: w_task = 0.70, w_speed = 0.30
        - `balanced`: w_task = 0.50, w_speed = 0.50
        - `speed`: w_task = 0.30, w_speed = 0.70
      - Final score calculation incorporating harmonized piecewise RAM penalty: `(w_task * task_fit + w_speed * speed_score) - evaluation.ram_penalty`.
      - Deterministic sorting by `final_score` descending and 1-based rank assignment.
    - `display_scorecard_table()`: Rich terminal table displaying rank, model repository, quant method, estimated speed, RAM requirement, budget ceiling, status badge, and final score.
12. **Resumable Chunked Downloader with SHA-256 Verification (`src/companion_packager/installer/downloader.py`):**
    - `download_file()`:
      - Idempotency: Checks existing destination file with `compute_sha256`; skips download if hash matches expected.
      - Pre-flight check: Calls `check_disk_space_preflight(dest_path.parent, required_model_bytes=estimated_size, is_installer_build=False)`. Raises `DiskSpaceError` on insufficient space.
      - `.part` staging: Reads existing bytes from `{dest}.part`, sends `Range: bytes={existing_bytes}-`.
      - HTTP 206 Partial Content: Appends incoming chunks in `"ab"` mode.
      - HTTP 200 OK: Truncates `.part` to 0 bytes and streams full content from beginning.
      - HTTP 416 Range Not Satisfiable: Verifies if file is already complete by checking Content-Range total, expected size, or SHA-256; if complete, proceeds to finalization; else truncates `.part` and restarts stream.
      - Integrity check: Computes full streaming SHA-256 on `.part`; renames to `.corrupt` and raises `DownloadChecksumError` on mismatch.
      - Atomic finalization: Renames verified `.part` to destination file.
      - `rich.progress.Progress`: Renders transfer speed, byte counts, and percentage when `show_progress=True`.
13. **Runtime Configuration Injector & Path Resolver (`src/companion_packager/installer/configurator.py`):**
    - `inject_agent_companion_config()`:
      - Generates `agent_companion.env` containing model path, backend device, context window, SSE host/port, physical thread count, and quant method.
      - Generates `start_companion.py`: native Python launcher with custom `.env` parser stripping surrounding quotes, setting `os.environ`, and supervising `uvicorn.run("security_server.server:app")`.
      - Generates `start_companion.bat`: Windows batch script using `for /F "usebackq eol=# tokens=1* delims==" %%i in ... set "%%i=%%~j"` for robust quote stripping.
      - Generates `start_companion.sh`: POSIX shell script with executable permissions and clean LF newlines.
    - `resolve_companion_path()`: Deterministic search order (CLI flag -> `AGENT_COMPANION_PATH` env var -> `../Agent Companion` -> `../../Agent Companion` -> `D:\Projects\Agent Companion`).
    - `create_desktop_shortcut()`: Fail-soft desktop shortcut creation for Windows PowerShell COM and Linux desktop entry.
14. **Isolated Monorepo Wheel Staging & Compilation (`src/companion_packager/installer/wheel_vendor.py`):**
    - `stage_and_build_monorepo_wheels()`: Uses `tempfile.TemporaryDirectory()` to stage packages, injects stub `README.md` strictly in temporary staging when missing (preventing hatchling failure while keeping source checkout clean), and executes `python -m build --wheel --no-isolation` with `hatchling`.
    - `build_packager_wheel()`: Builds `companion-packager` itself into a wheel and places it into `wheels/`.
15. **Target ABI Matching & Dependency Filtering (`src/companion_packager/installer/wheel_vendor.py`):**
    - `extract_and_filter_requirements()`: Uses stdlib `tomllib` to parse `pyproject.toml` files, filters out local workspace packages (`shared-contracts`, `model-runtime`, `security-server`, `document-pipeline`, `agent-tooling`, `companion-packager`), and appends packager dependencies. Returns sorted unique requirements list.
    - `vendor_third_party_wheels()`: Supports offline cache copying and online `pip download` targeting Python 3.11 ABI (`--python-version 3.11 --platform <target> --abi cp311 --only-binary=:all: --implementation cp`). Raises `ValueError` when offline without cache.
16. **Pinned Python Standalone Acquisition (`src/companion_packager/installer/wheel_vendor.py`):**
    - `acquire_python_standalone()`: Acquires pinned release `20241016` (`install_only.tar.gz`) from `astral-sh/python-build-standalone` for Windows (`x86_64-pc-windows-msvc-shared`), Linux (`x86_64-unknown-linux-gnu`), and macOS (`aarch64-apple-darwin`).
    - Strips leading root folder during extraction so binaries reside directly at `runtime/python.exe` or `runtime/bin/python3`.
17. **Inno Setup Windows Compiler & Portable Fallback (`src/companion_packager/installer/inno_builder.py`):**
    - `generate_inno_script()`: Generates valid `.iss` script with `#define SourceDir`, `PrivilegesRequired=lowest`, `DefaultDirName={localappdata}\Programs\Agent Companion`, `[Files]`, `[Icons]`, and `[Run]` invoking offline pip install and companion launcher.
18. **Typer CLI Dynamic Search & Harvester Options (`src/companion_packager/cli.py`):**
    - Extended `analyze` and `recommend` subcommands with `--live / --no-live` (default: True), `--search / -s <query>`, and `--limit <count>`.
    - Connected options to `get_model_candidates` for live dynamic discovery.

---

## Files Created, Modified, or Removed
- `src/companion_packager/research/hf_client.py`: Implemented `parse_param_count_from_name`, `detect_quant_and_file`, `extract_arxiv_and_doi`, `extract_license`, `estimate_model_layers`, `quant_to_bits_per_weight`, `discover_trending_models`, and updated `get_model_candidates`.
- `src/companion_packager/research/__init__.py`: Exported dynamic discovery utilities.
- `src/companion_packager/cli.py`: Extended `_run_analysis`, `analyze`, and `recommend` with `--live`, `--no-live`, `--search`, and `--limit`.
- `tests/test_hf_discovery.py`: Dedicated test suite covering regex extraction, quant priority, arXiv parsing, layer estimation, mocked discovery, and CLI search.
- `tests/test_cli.py`: Added tests for `--search`, `--live`, and `--no-live` options.
- `.agent/PROGRESS.md`: Updated with Phase 8 discovery engine progress, verification logs, and artifacts.

---

## Test Results & Verification Logs
- `pytest -v tests/test_hf_discovery.py`:
  - `test_parse_param_count_from_name`: PASSED
  - `test_detect_quant_and_file_priorities`: PASSED
  - `test_extract_arxiv_and_doi`: PASSED
  - `test_extract_license`: PASSED
  - `test_estimate_model_layers`: PASSED
  - `test_discover_trending_models_mocked`: PASSED
  - `test_discover_trending_models_network_error_graceful`: PASSED
  - `test_get_model_candidates_deduplication`: PASSED
  - `test_cli_analyze_search`: PASSED
- `pytest -v tests/test_cli.py`: 13/13 PASSED
- Full Test Suite: `pytest -v tests/` -> 124/124 passed in 33.14s (100% pass rate).
- Live CLI Verification:
  - `companion-packager analyze --search "coder" --headless --limit 5`: PASSED. Successfully queried live Hugging Face Hub, parsed GGUF candidates, extracted metadata/quant, computed sizing and throughput, and ranked candidates without error.
  - `companion-packager analyze --search "deepseek" --headless --limit 3`: PASSED.
- Wheel Build:
  - `python -m build --wheel` -> `companion_packager-0.1.0-py3-none-any.whl` produced cleanly in `dist/`.

---

## Orchestrator Handoff Context
- Dynamic Hugging Face Discovery Engine and live leaderboard harvester is completely implemented and tested.
- 124 automated tests passing across 14 test suites.
- Distributable wheel is built and ready in `dist/`.

---

## Active Gotchas & Failed Paths
- `HfApi.list_models()` in modern `huggingface_hub` supports `sort="trending_score"` (not `trending` or `direction=-1`). Handled defensively with cascading fallbacks to support both current remote API and legacy mock signatures.
- Remote models on Hugging Face Hub may omit LFS file size attributes in metadata; `quant_to_bits_per_weight` provides exact theoretical bits-per-weight fallback to ensure physically sound sizing calculations.

---

## Next Immediate Action
- Hand off to Elephant Orchestrator.
