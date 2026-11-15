# Agent Companion Packager - Progress & Tracking

## Current Active Phase & Status
- **Current Phase:** Phase 4 Complete: Interactive Questionnaire, Sizing Matrix & Multi-Backend Recommender Engine
- **Status:** PASSED. Interactive Rich-prompt and headless user needs interview (`ask_user_needs`), multi-tier candidate recommendation and ranking matrix (`rank_candidates`), permissive license filtering, context length enforcement, hardware profile sizing integration, and Rich terminal scorecard table renderer (`display_scorecard_table`) implemented and verified with 100% test pass rate (67/67 tests passing across entire suite).
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

---

## Files Created, Modified, or Removed
- `src/companion_packager/interview/__init__.py`: Package init exporting `ask_user_needs`.
- `src/companion_packager/interview/questionnaire.py`: Interactive Rich and headless user needs interview.
- `src/companion_packager/sizing/recommender.py`: Candidate model recommendation, ranking matrix, and scorecard table renderer.
- `src/companion_packager/sizing/__init__.py`: Updated exports for `rank_candidates`, `display_scorecard_table`, and `PERMISSIVE_LICENSES`.
- `src/companion_packager/sizing/formulas.py`: Enhanced `is_mla` check to recognize `deepseek_v2` and `deepseek` architectures.
- `tests/test_recommender.py`: 11 unit tests covering headless defaults, explicit flags, validation errors, monkeypatched interactive prompts, permissive license filtering, sub-6GB CPU laptop invariants, MoE mid-tier GPU speedup, high-tier 24GB+ GPU quality ranking, priority shifts, context window violations, and Rich table rendering.
- `.agent/PROGRESS.md`: Updated with Phase 4 status and test verification logs.

---

## Test Results & Verification Logs
- `pytest -v tests/test_recommender.py`:
  - `test_ask_user_needs_headless_defaults`: PASSED (defaults validated).
  - `test_ask_user_needs_headless_custom`: PASSED (explicit arguments respected).
  - `test_ask_user_needs_validation_errors`: PASSED (invalid task, priority, context rejected).
  - `test_ask_user_needs_interactive_mocked`: PASSED (4 interactive prompts captured).
  - `test_permissive_license_filtering`: PASSED (DeepSeek excluded under permissive filter, included otherwise).
  - `test_cpu_laptop_profile_recommendation`: PASSED (Qwen 7B fits under 6144 MB ceiling and ranks #1 for coding; 14B and 32B rejected with -900+ penalty).
  - `test_mid_tier_gpu_recommendation`: PASSED (DeepSeek-Coder-V2-Lite MoE active_weight_ratio 0.15 gives >200 tok/s and top tier rank).
  - `test_high_tier_24gb_gpu_recommendation`: PASSED (Qwen 32B achieves 0.978 task fit and ranks #1 for quality).
  - `test_priority_weighting_shifts`: PASSED (quality vs speed priority weight shifts verified).
  - `test_context_window_violation_penalized`: PASSED (8k context model rejected when 16k context requested).
  - `test_display_scorecard_table`: PASSED (Rich table rendered without exceptions).
- Full Test Suite: `pytest -v tests/` -> 67/67 passed in 5.04s.
- Wheel Build: `python -m build --wheel` -> `companion_packager-0.1.0-py3-none-any.whl` built cleanly.

---

## Orchestrator Handoff Context
- Phase 4 complete. Interactive interview (`ask_user_needs`), scoring matrix (`rank_candidates`), and scorecard visualizer (`display_scorecard_table`) operational and fully verified.
- Sub-6GB RAM invariant verified on CPU profiles; MoE 0.15 active weight throughput verified on GPU profiles; license and context length constraints verified.
- Ready for Phase 5: Resumable Model Downloader & Agent Companion Configurator (`downloader.py`, `configurator.py`).

---

## Active Gotchas & Failed Paths
- None. When evaluating 32B on workstation GPUs, VRAM envelope must accommodate total runtime memory (weights + KV cache + overhead) without exceeding 87.5% threshold to avoid piecewise RAM penalty.

---

## Next Immediate Action
- Hand off to Orchestrator for Phase 5 dispatch (Resumable downloader with HTTP range/resume/SHA-256 verification, and Agent Companion configuration generator).
