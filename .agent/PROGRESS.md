# Agent Companion Packager - Progress & Tracking

## Current Active Phase & Status
- **Current Phase:** Phase 6 Complete: Recursive Wheel Vendor & Cross-Platform Packaging Engine
- **Status:** PASSED. Isolated monorepo wheel builder (`stage_and_build_monorepo_wheels`), workspace package filtering & dependency merger (`extract_and_filter_requirements`), pinned Python standalone runtime acquirer (`acquire_python_standalone`), third-party wheel vendorizer with target ABI matching (`vendor_third_party_wheels`), Inno Setup compiler & portable ZIP packager (`inno_builder.py`), and POSIX self-extracting archive builder (`sfx_builder.py`) implemented and verified with 100% test pass rate (98/98 tests passing across entire suite).
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

---

## Files Created, Modified, or Removed
- `src/companion_packager/installer/wheel_vendor.py`: Monorepo wheel builder, workspace dependency filter, standalone Python acquirer, and third-party wheel vendorizer.
- `src/companion_packager/installer/inno_builder.py`: Inno Setup script generator and Windows installer / portable ZIP packager.
- `src/companion_packager/installer/sfx_builder.py`: POSIX self-extracting archive builder (.run) for Linux and macOS.
- `src/companion_packager/installer/__init__.py`: Exported Phase 6 functions.
- `tests/test_wheel_vendor.py`: 6 unit tests covering dependency extraction & workspace filtering, isolated staging without source mutation, standalone Python root stripping, and offline/online wheel vendoring.
- `tests/test_inno_builder.py`: 3 unit tests covering Inno Setup directives, portable ZIP fallback, and compiler execution.
- `tests/test_sfx_builder.py`: 2 unit tests covering POSIX SFX bash header, sed extraction, tar payload integrity, and custom app naming.
- `.agent/PROGRESS.md`: Updated with Phase 6 status and test verification logs.

---

## Test Results & Verification Logs
- `pytest -v tests/test_wheel_vendor.py`:
  - `test_extract_and_filter_requirements_mock`: PASSED
  - `test_extract_and_filter_requirements_real_repo`: PASSED
  - `test_stage_and_build_monorepo_wheels_isolated_staging`: PASSED
  - `test_acquire_python_standalone_root_stripping`: PASSED
  - `test_vendor_third_party_wheels_cache_and_offline`: PASSED
  - `test_vendor_third_party_wheels_online_invocation`: PASSED
- `pytest -v tests/test_inno_builder.py`:
  - `test_generate_inno_script_directives`: PASSED
  - `test_build_windows_installer_fallback_to_portable_zip`: PASSED
  - `test_build_windows_installer_with_mock_iscc`: PASSED
- `pytest -v tests/test_sfx_builder.py`:
  - `test_build_posix_sfx_structure`: PASSED
  - `test_build_posix_sfx_custom_app_name`: PASSED
- Full Test Suite: `pytest -v tests/` -> 98/98 passed in 8.40s.
- Wheel Build: `python -m build --wheel` -> `companion_packager-0.1.0-py3-none-any.whl` built cleanly.

---

## Orchestrator Handoff Context
- Phase 6 complete. All wheel vendorizing, Python standalone acquisition, Inno Setup script / portable packaging, and POSIX SFX archive generation implemented and verified.
- Monorepo staging leaves source checkout 100% clean, workspace packages are cleanly filtered from requirements, and Python standalone runtime extracts with root prefixes stripped.
- Inno Setup scripts and portable ZIP fallback validated; POSIX SFX archives with `sed` payload extraction validated.
- Ready for Phase 7: End-to-End System Test & Verification across CLI commands (`profile`, `recommend`, `download`, `build`, `install`).

---

## Active Gotchas & Failed Paths
- In `sfx_builder.py`, `sed -e '1,/^__ARCHIVE_BELOW__/d' "$0"` expects `__ARCHIVE_BELOW__\n` as the exact boundary separating the bash script header from binary tar stream. Script header uses strictly LF line endings.

---

## Next Immediate Action
- Hand off to Orchestrator for Phase 7 dispatch (End-to-end integration test across entire pipeline: Hardware scan -> Needs interview -> HF research -> Sizing decision -> Config generation -> Wheel vendor -> Installer build).
