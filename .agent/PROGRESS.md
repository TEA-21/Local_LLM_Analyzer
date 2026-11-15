# Agent Companion Packager - Progress & Tracking

## Current Active Phase & Status
- **Current Phase:** Phase 3 Complete: Model Research, Hugging Face Client, Benchmark Registry & Citations
- **Status:** PASSED. 100% offline airgapped fallback benchmark dataset, two-way benchmark imputation, robust BenchNorm calculations, Hugging Face Hub LFS metadata queries with gated error handling, structured citation cards, and comprehensive RECOMMENDATIONS.md markdown synthesis implemented and verified with 100% test pass rate (56/56 tests passing across entire suite).
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

---

## Files Created, Modified, or Removed
- `src/companion_packager/research/__init__.py`: Package init exporting public API.
- `src/companion_packager/research/benchmarks_cache.json`: Airgapped candidate models registry.
- `src/companion_packager/research/benchmarks.py`: Benchmark registry, two-way imputation, and normalization engine.
- `src/companion_packager/research/hf_client.py`: Resilient HF Hub client with offline fallback.
- `src/companion_packager/research/citations.py`: Structured citation card and RECOMMENDATIONS.md generator.
- `tests/test_research.py`: 10 comprehensive unit tests covering offline cache loading, two-way imputation, BenchNorm bounds, HF client online LFS size updates, gated repo fallback, citation cards, and markdown report synthesis.
- `.agent/PROGRESS.md`: Updated with Phase 3 status and test verification logs.

---

## Test Results & Verification Logs
- `pytest -v tests/test_research.py`:
  - `test_load_benchmarks_cache_success`: PASSED (7 candidate models validated against Pydantic schemas).
  - `test_load_benchmarks_cache_missing_file`: PASSED (FileNotFoundError handled cleanly).
  - `test_normalize_benchmarks_imputation`: PASSED (SWE-bench <-> HumanEval and Arena-Elo <-> OpenLLM two-way imputation verified).
  - `test_normalize_benchmarks_weights_and_bounds`: PASSED (4-term formula and [0.0, 1.0] bounding verified).
  - `test_normalize_benchmarks_task_domains`: PASSED (coding, reasoning, and chat domain formulas verified).
  - `test_hf_client_offline_mode`: PASSED (100% offline airgapped operation verified).
  - `test_hf_client_online_lfs_size_update`: PASSED (sibling LFS size update verified).
  - `test_hf_client_gated_and_network_error_fallback`: PASSED (gated repo and network failure fallback to cache verified).
  - `test_generate_citation_card`: PASSED (simple and sizing-evaluation-augmented cards verified).
  - `test_generate_recommendations_markdown`: PASSED (full RECOMMENDATIONS.md artifact structure verified).
- Full Test Suite: `pytest -v tests/` -> 56/56 passed in 5.47s.
- Wheel Build: `python -m build --wheel` -> `companion_packager-0.1.0-py3-none-any.whl` built cleanly, containing `companion_packager/research/benchmarks_cache.json`.

---

## Orchestrator Handoff Context
- Phase 3 complete. Research subsystem, offline benchmark registry, HF Hub client, and citation synthesizer operational and fully verified.
- All candidate models deserialize into Pydantic `CandidateModel` schemas with verified benchmarks and DOIs.
- Ready for Phase 4: Sizing & Scoring Matrix (`src/companion_packager/sizing/matrix.py`, `questionnaire.py`, `recommender.py`).

---

## Active Gotchas & Failed Paths
- None. BenchNorm calculation rounds to 6 decimal places before clamping to avoid floating-point representation artifact (`0.9999999999999999` vs `1.0`).

---

## Next Immediate Action
- Hand off to Orchestrator for Phase 4 dispatch (Questionnaire, scoring matrix, and recommendation engine).
