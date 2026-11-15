# Agent Companion Packager - Progress & Tracking

## Current Active Phase & Status
- **Current Phase:** Phase 8 Complete: Multi-Domain Intent Query Planner & 3-Pillar Proof Citation Engine
- **Status:** All Phase 8 components implemented, integrated, and verified with 136/136 passing tests. Wheel build successful.
- **argon_audit_runs:** 1

---

## Completed Steps & Exact Techniques/Libraries Used
1. **Packaging & Scaffolding (`pyproject.toml`):**
   - Configured `hatchling.build` backend, Python >= 3.11, entrypoint `companion-packager = "companion_packager.cli:app"`.
   - Declared core dependencies: `typer>=0.12.0`, `rich>=13.7.0`, `pydantic>=2.7.0`, `psutil>=5.9.8`, `py-cpuinfo>=9.0.0`, `huggingface-hub>=0.23.0`, `httpx>=0.27.0`, `build>=1.2.0`, `hatchling>=1.21.0`.
   - Verified wheel building with `python -m build --wheel` produces `companion_packager-0.1.0-py3-none-any.whl`.
2. **Pydantic v2 Domain Schemas (`src/companion_packager/schemas.py` & `models/schemas.py`):**
   - Implemented `GPUInfo`, `HardwareProfile`, `UserNeeds`, `CandidateModel`, `SizingEvaluation`, `Scorecard`, and `ProofCitation`.
   - Dual-exported via `companion_packager.schemas` and `companion_packager.models.schemas`.
3. **Mathematical Sizing & Throughput Formulas (`src/companion_packager/sizing/formulas.py`):**
   - `calculate_weights_mb`: Exact parameter count and bits-per-weight equation.
   - `calculate_kv_cache_mb`: Standard GQA/MHA and DeepSeek MLA attention formulations with FP16 and Q8_0 precision fallback.
   - `calculate_budget_ceiling_mb`: GPU offloading formula `min(total * 0.80, free * 0.95)` and hard CPU memory invariant `min(avail_ram * 0.80, 6144.0 MB)`.
   - `estimate_tokens_per_second`: Memory bandwidth bound throughput with MoE `active_weight_ratio` scaling, clamped sparsity factor `[0.0, 1.0]`, lower-bounded `effective_ratio = max(active_weight_ratio, 0.05)`, and safe denominator `max(weights_gb * effective_ratio, 0.001)`.
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
   - Bundles 15 verified candidate models stratified across hardware envelopes and domains:
     - General Desktop: Llama-3.2 3B (2050 MB), SmolLM2 1.7B (1050 MB).
     - Creative Writing: Meta-Llama-3.1 8B (4920 MB), Mistral-7B v0.3 (4370 MB).
     - Summarization: Qwen2.5 7B (4680 MB), Qwen2.5 3B (2050 MB).
     - Reasoning & Math: DeepSeek-R1-Distill-Qwen 1.5B (805 MB), DeepSeek-R1-Distill-Qwen 7B (4680 MB).
     - Sub-6GB CPU Coding: Qwen2.5-Coder 7B (4680 MB), 3B (2050 MB), 1.5B (1150 MB).
     - Mid-to-High Tier GPU: DeepSeek-Coder-V2-Lite MoE (9600 MB, active ratio 0.15, MLA), Qwen2.5-Coder 14B (9200 MB), TernaryQuench-Qwen3.8-27B Q2_K (10360 MB), Qwen2.5-Coder 32B (19800 MB).
   - Packaged directly inside wheel distribution for 100% airgapped runtime.
7. **Connected Bidirectional Cross-Imputation Engine (`src/companion_packager/research/benchmarks.py`):**
   - Unified graph connecting Island A (SWE-bench, HumanEval, GSM8K, MATH) and Island B (OpenLLM_v2, MMLU-Pro, Arena_Elo, IFEval, EQ-Bench, MT-Bench_Writing) via `GSM8K <-> OpenLLM_v2` bridge.
   - Standardized `calculate_elo_norm(arena_elo)` in `[0.0, 1.0]`.
   - `impute_missing_benchmarks`: Strict multi-source resolution precedence (tier 0 direct measured values first, followed by arithmetic means of 1-hop imputations, then 2-hop).
   - `normalize_benchmarks`: Evaluates BenchNorm across all 6 domains (`coding`, `creative_writing`, `general_desktop`, `reasoning_math`, `summarization`, `custom`) with backward compatibility for legacy tasks.
8. **Multi-Domain Intent Query Planner & Harvester (`src/companion_packager/research/hf_client.py`):**
   - Configured `TASK_QUERY_BUNDLES` covering all workload domains and custom freeform parsing.
   - `discover_trending_models` and `get_model_candidates`: Accept `task` and `custom_intent` to dynamically query Hugging Face Hub while deduplicating against airgapped cache.
9. **Questionnaire & Intent Tokenizer (`src/companion_packager/interview/questionnaire.py`):**
   - `parse_custom_intent`: Extracts top 3 domain keywords with support for programming languages and symbols (`c++`, `c#`, `sql`, `rag`), stopword stripping, and frequency/length sorting.
   - `ask_user_needs`: Supports all multi-domain tasks and headless/interactive custom intent handling.
10. **3-Pillar Proof Citation Engine (`src/companion_packager/research/citations.py`):**
    - `generate_proof_citation`: Assembles Pillar 1 (task benchmark score with `[Measured]` / `[Estimated]` provenance), Pillar 2 (runtime memory allocation, headroom safety margin, tok/s throughput), and Pillar 3 (plain-language "Why This Model" justification bullets).
    - `generate_proof_citation_card`: Formats structured markdown card.
    - `generate_recommendations_markdown`: Synthesizes complete report embedding 3-pillar proof cards for all ranked candidates.
11. **Sizing Recommender & Hardware Gating (`src/companion_packager/sizing/recommender.py`):**
    - Implemented `hard_exclude` option filtering models whose weights exceed budget ceiling on constrained hosts.
    - Attached `ProofCitation` to each generated `Scorecard`.
12. **CLI Multi-Domain Commands (`src/companion_packager/cli.py`):**
    - Updated `--task` with choices across all 6 domains.
    - Added `--custom-intent` option to `analyze` and `recommend` commands.

---

## Files Created, Modified, or Removed
- **Modified:**
  - `src/companion_packager/schemas.py`
  - `src/companion_packager/models/schemas.py`
  - `src/companion_packager/sizing/formulas.py`
  - `src/companion_packager/research/benchmarks_cache.json`
  - `src/companion_packager/research/benchmarks.py`
  - `src/companion_packager/research/__init__.py`
  - `src/companion_packager/interview/questionnaire.py`
  - `src/companion_packager/research/hf_client.py`
  - `src/companion_packager/research/citations.py`
  - `src/companion_packager/sizing/recommender.py`
  - `src/companion_packager/cli.py`
  - `tests/test_hf_discovery.py`
  - `tests/test_research.py`
- **Created:**
  - `tests/test_domain_proof.py`

---

## Test Results & Verification Logs
- `pytest -v tests/test_domain_proof.py`: 12/12 PASSED in 0.12s
  - `test_parse_custom_intent_languages_and_symbols`: PASSED
  - `test_parse_custom_intent_rag_and_stopwords`: PASSED
  - `test_parse_custom_intent_empty_and_fallback`: PASSED
  - `test_ask_user_needs_multi_domain_and_custom`: PASSED
  - `test_imputation_island_bridge_coding_to_desktop`: PASSED
  - `test_imputation_island_bridge_desktop_to_coding`: PASSED
  - `test_imputation_precedence_and_multi_source_mean`: PASSED
  - `test_calculate_elo_norm_bounds`: PASSED
  - `test_domain_benchnorm_formulas`: PASSED
  - `test_proof_citation_generation_and_provenance`: PASSED
  - `test_hardware_ceiling_cpu_gating`: PASSED
  - `test_estimate_tokens_per_second_clamping_and_bounds`: PASSED
- `pytest -v tests/test_research.py`: 10/10 PASSED
- `pytest -v tests/test_recommender.py`: 11/11 PASSED
- `pytest -v tests/test_hf_discovery.py`: 9/9 PASSED
- Full Test Suite: `pytest -v tests/` -> 136/136 passed in 28.54s (100% pass rate).
- Wheel Build: `python -m build --wheel` -> `companion_packager-0.1.0-py3-none-any.whl` produced cleanly in `dist/`.

---

## Orchestrator Handoff Context
- Phase 8 Multi-Domain Intent Query Planner & 3-Pillar Proof Citation Engine is fully implemented and tested.
- All 136 automated tests across 15 test suites are green.
- Airgapped cache contains 15 verified candidate models covering all 5 core workload domains.
- Distributable wheel is compiled in `dist/`.

---

## Active Gotchas & Failed Paths
- In `test_get_model_candidates_deduplication`, the previous hardcoded assertion checked for 8 models (7 cached + 1 new); updated to dynamically assert `len(load_benchmarks_cache()) + 1` to accommodate multi-domain cache expansion.
- In `normalize_benchmarks`, default parameter `task=None` preserves the default 4-term BenchNorm formula for general requests while allowing explicit domain formulas for specific workloads.

---

## Next Immediate Action
- Hand off to Elephant Orchestrator with summary and verification results.
