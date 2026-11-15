# Agent Companion Packager - Progress & Tracking

## Current Active Phase & Status
- **Current Phase:** Phase 7 Complete: Modular Typer CLI, End-to-End Integration Test Harness & Distributable Wheel
- **Status:** PASSED. Modular Typer CLI application (`src/companion_packager/cli.py` & `__main__.py`) implementing commands (`version`, `doctor`, `profile`, `analyze`, `recommend`, `download`, `build`, `install`), unit test suite (`tests/test_cli.py`), and full E2E user lifecycle integration suite (`tests/test_e2e_integration.py`) implemented and verified with 100% test pass rate (113/113 tests passing across entire suite in 8.11s). Distributable wheel `companion_packager-0.1.0-py3-none-any.whl` compiled and verified.
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
8. **Hugging Face Hub Client with Gated Repo Fallback (`src/companion_packager/research/hf_client.py`):**
   - `get_model_candidates()`: When online, queries `api.model_info(repo_id, files_metadata=True)` to verify file sizes against LFS metadata.
   - Defensively catches connection errors, timeouts, and HTTP 401/403 gated model errors (`HfHubHTTPError`, `GatedRepoError`), falling back cleanly to cached candidates.
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
    - `build_windows_installer()`: Probes host for `iscc.exe` (PATH, Program Files, LocalAppData). If found, compiles `AgentCompanion-Installer.exe`. If missing, falls back to generating `AgentCompanion-Windows-Portable.zip` bundling all runtime files and launchers (`run_setup.bat`, `install.bat`, `start_companion.bat`).
18. **POSIX Self-Extracting Archive Builder (`src/companion_packager/installer/sfx_builder.py`):**
    - `build_posix_sfx()`: Compresses bundle directory into in-memory `tar.gz`, prepends bash self-extracting header with `sed` payload extraction, offline pip installation, launcher script chmodding, and launcher symlinking (`$HOME/.local/bin/agent-companion`). Sets executable permission bits (`0o755`).
19. **Modular Typer CLI Application (`src/companion_packager/cli.py` & `__main__.py`):**
    - `app = typer.Typer(name="companion-packager")`: Complete command-line application with entrypoint in `__main__.py`.
    - `version`: Prints `companion-packager v0.1.0` (also supports eager `--version` / `-v` flag).
    - `doctor`: Comprehensive host sanity audit inspecting Python >= 3.11, CPU SIMD extensions (AVX-512, AVX2, NEON), RAM vs 6144 MB ceiling invariant, zero-PyTorch GPU detection, disk space, and Agent Companion packages.
    - `profile`: Outputs host hardware profile as formatted Rich table or raw JSON (`--json`).
    - `analyze`: End-to-end analyzer supporting `--task`, `--priority`, `--context-len`, `--license`, `--interactive/--headless`, `--offline`, `--output-md`, and `--sparsity-speedup`. Renders scorecard table and top model citation card.
    - `recommend`: CLI alias for streamlined headless recommendations.
    - `download`: Resumable chunked model downloader with `--dest`, `--offline`, `--expected-sha256`, and Rich progress display.
    - `build`: Full installer compilation supporting `--mode thin|fat`, `--format installer|portable|sfx`, disk preflight checks, monorepo staging, wheel vendoring, standalone runtime acquisition, configuration injection, and portable/Inno/SFX packaging.
    - `install`: Target machine offline setup extracting bundles, running offline pip install, and creating desktop shortcuts.
20. **Comprehensive Unit & E2E Integration Test Suite (`tests/test_cli.py` & `tests/test_e2e_integration.py`):**
    - Added 11 unit tests in `test_cli.py` verifying all CLI command invocations, options, table formatting, and headless executions.
    - Added 4 comprehensive E2E integration tests in `test_e2e_integration.py` verifying the full user lifecycle: hardware probing -> needs discovery -> offline model sizing -> bundle staging -> config generation (`agent_companion.env`, `start_companion.bat`, `start_companion.sh`, `start_companion.py`) -> quote-stripping batch verification -> portable installer zip compilation -> archive integrity -> offline target install.

---

## Files Created, Modified, or Removed
- `src/companion_packager/cli.py`: Complete Typer CLI application implementing all subcommands.
- `src/companion_packager/__main__.py`: CLI entrypoint module.
- `tests/test_cli.py`: Unit tests for CLI commands and argument options (11 tests).
- `tests/test_e2e_integration.py`: End-to-end integration tests simulating full user workflows (4 tests).
- `.agent/PROGRESS.md`: Updated with Phase 7 status, test logs, and artifact locations.

---

## Test Results & Verification Logs
- `pytest -v tests/test_cli.py`:
  - `test_cli_version`: PASSED
  - `test_cli_version_flag`: PASSED
  - `test_cli_doctor`: PASSED
  - `test_cli_profile_table`: PASSED
  - `test_cli_profile_json`: PASSED
  - `test_cli_analyze_headless`: PASSED
  - `test_cli_recommend_alias`: PASSED
  - `test_cli_download_offline_exists`: PASSED
  - `test_cli_download_with_mock`: PASSED
  - `test_cli_build_help`: PASSED
  - `test_cli_install_help`: PASSED
- `pytest -v tests/test_e2e_integration.py`:
  - `test_e2e_user_workflow_cpu_pipeline`: PASSED
  - `test_e2e_user_workflow_gpu_pipeline`: PASSED
  - `test_e2e_cli_install_flow`: PASSED
  - `test_e2e_cli_build_thin_mode`: PASSED
- Full Test Suite: `pytest -v tests/` -> 113/113 passed in 8.11s.
- CLI Invocation:
  - `python -m companion_packager version` -> `companion-packager v0.1.0`
  - `python -m companion_packager --help` -> Help menu with 8 commands rendered.
- Wheel Build:
  - `python -m build --wheel` -> `companion_packager-0.1.0-py3-none-any.whl` produced cleanly in `dist/`.

---

## Orchestrator Handoff Context
- All 7 phases of the master specification in `.agent/PLAN.md` are 100% implemented, integrated, and verified with zero regressions.
- Test coverage expanded to 113 automated tests across 13 test suites.
- Specification drift audit confirmed zero unapproved changes, zero orphaned files, and strict adherence to architectural contracts.
- Ready for final Orchestrator review and sign-off.

---

## Active Gotchas & Failed Paths
- In Typer, direct programmatic invocation of command functions with defaults requires separating command handlers from underlying execution helpers (`_run_analysis`) to avoid passing `OptionInfo` objects as runtime parameters.

---

## Next Immediate Action
- Hand off to Elephant Orchestrator for final specification drift audit and project wrap-up.
