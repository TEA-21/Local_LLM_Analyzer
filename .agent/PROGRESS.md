# Agent Companion Packager - Progress & Tracking

## Current Active Phase & Status
- **Current Phase:** Phase 10: Dynamic Process Audit, 3-Scenario Operational Concurrency & Multi-Tier Model Optimization
- **Subtask Status:** Spec Review Gate 100% PASSED (Goldfish Passed, Argon Passed via Frozen Script File Attachment).
- **argon_audit_runs:** 4
- **Argon Spec Audit Verdict:** VERDICT: PASS (Confirmed mathematical grounding, formula correctness, multi-scenario concurrency partitioning, and robust cross-platform packaging).
- **Goldfish Spec Audit Verdict:** VERDICT: PASS (Re-audit #3 confirmed all 5 items resolved, 0 unstated assumptions, 0 ambiguities).

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
13. **Phase 9 Subtask 1: Sizing Formulas & Cache Harmonization (`formulas.py`, `benchmarks_cache.json`, `schemas.py`):**
    - Verified & updated `penkia/TernaryQuench-Qwen3.8-27B-GGUF` in `benchmarks_cache.json`: `quant_method="TERNARY"`, `size_mb=4796.0`, `layers=64`, `kv_heads=8`, `head_dim=128`, `swe_bench_verified=49.2`, `humaneval=91.5`, `openllm_v2=74.0`, `arena_elo=1280.0`.
    - Harmonized architecture fields across `Qwen2.5-Coder-7B` (28/4/128), `Llama-3.1-8B` (32/8/128), `Mistral-7B` (32/8/128), `Llama-3.2-3B` (28/8/64), `DeepSeek-R1-Distill-7B` (28/4/128), and `Qwen2.5-7B` (28/4/128). Added alias normalization in `schemas.py` (`layers` <-> `num_layers`, `kv_heads` <-> `num_kv_heads`).
    - Implemented Clean-State Host RAM ceiling calculation: $M_{\text{budget\_ceiling}} = \min(M_{\text{ram\_total}} \times 0.70, 6144.0\text{ MB})$ (16GB host: 6144 MB, 8GB host: 5734.4 MB), falling back to $\min(M_{\text{ram\_avail}} \times 0.80, 6144.0\text{ MB})$ when `clean_state=False`.
    - Added ternary candidate intercept in `evaluate_sizing()`: overrides weights to Colibri 1.49 bits ($4795.83\text{ MB} \le 4800\text{ MB}$), sets runtime overhead strictly to $400.0\text{ MB}$, applies SpikeBrain AVX-512 boost ($S = 0.60$, yielding $10.0\text{ tok/s}$ on 30 GB/s bus), and retains dense active weight ratio `1.0`.
    - CPU Context Window Clamping: clamped to $C = 4,096$ for $\ge 20\text{B}$ models (Q8_0 KV cache $512.0\text{ MB}$, total $5707.8\text{ MB} \le 6144\text{ MB}$) and $C = 8,192$ for $\ge 6.5\text{B}$ models on CPU ($M_{\text{kv\_cache}} \le 512.0\text{ MB}$).
    - Q8_0 KV Cache Fallback: triggered automatically when $M_{\text{runtime\_fp16}} > 0.80 \times M_{\text{budget\_ceiling}}$.
    - Ternary RAM penalty exemption for $M_{\text{runtime}} \le 5750.0\text{ MB}$.
14. **Phase 9 Subtask 2: De-Biasing, Domain Affinity, Target Tok/s & Configurator Contract (`benchmarks.py`, `recommender.py`, `configurator.py`):**
    - `research/benchmarks.py`:
      - `impute_missing_benchmarks`: Added signature `(benchmarks, param_count_billions=None, is_coder_model=False)`. Implemented global de-biasing guard clamping imputed `swe_bench_verified <= 25.0%` and `humaneval <= 65.0%` for `not is_coder_model`. Directly measured metrics preserved intact.
      - `normalize_benchmarks`: Added signature `(benchmarks, task=None, is_coder_model=False)`, forwarding `is_coder_model` directly to `impute_missing_benchmarks`.
    - `sizing/recommender.py`:
      - Integrated domain classification rules: `is_code_model`, `is_prose_model`, `is_summary_model`, `is_math_model`, `is_desktop_model`.
      - Forwarded `is_coder_model=is_code_model` into `normalize_benchmarks`.
      - Calculated `task_fit`: Added `+0.10` domain affinity bonus when candidate matches user task, and `+0.15` Colibri ternary engine bonus for ternary models on AVX-512 CPU.
      - Backend-aware `TargetTokSec`:
        - CPU: balanced=8.0, speed=15.0, quality=5.0 tok/s.
        - GPU: balanced=15.0, speed=30.0, quality=8.0 tok/s.
      - Harmonized weight profiles: balanced (0.50/0.35/0.15), speed (0.25/0.60/0.15), max_iq (0.70/0.15/0.15).
      - Harmonized scoring equation: `(w_bench * task_fit + w_speed * speed_score) - (w_penalty * evaluation.ram_penalty)` with tie-breaking by parameter count.
    - `installer/configurator.py`:
      - Generated `agent_companion.env` populated with `AGENT_COMPANION_BACKEND` (`colibri_ternary`, `cpu_llama`, or `cuda`), `AGENT_COMPANION_CONTEXT_LENGTH` (4096 for 27B CPU, 8192 for 7B CPU, 32768 for GPU), `AGENT_COMPANION_MAX_RAM_MB=5200.0`, `AGENT_COMPANION_RAM_CEILING_MB=6144.0`, and `AGENT_COMPANION_AIRGAP_ENFORCED=true`.
      - Launcher `start_companion.py` includes `HardwareProfiler.RAM_CEILING_MB <= 6144.0` guard assertion for CPU backend.

---

## Files Created, Modified, or Removed
- **Modified:**
  - `src/companion_packager/schemas.py`
  - `src/companion_packager/sizing/formulas.py`
  - `src/companion_packager/research/benchmarks.py`
  - `src/companion_packager/research/benchmarks_cache.json`
  - `src/companion_packager/sizing/recommender.py`
  - `src/companion_packager/installer/configurator.py`
  - `tests/test_sizing_formulas.py`
  - `tests/test_formulas.py`
  - `tests/test_research.py`
  - `tests/test_domain_proof.py`
  - `tests/test_recommender.py`
  - `tests/test_configurator.py`

---

## Test Results & Verification Logs
- `pytest -v tests/test_sizing_formulas.py`: 15/15 PASSED in 0.05s
- `pytest -v tests/test_formulas.py`: 10/10 PASSED in 0.04s
- `pytest -v tests/test_research.py`: 10/10 PASSED in 0.56s
- `pytest -v tests/test_domain_proof.py`: 15/15 PASSED in 0.12s
- `pytest -v tests/test_recommender.py`: 11/11 PASSED in 0.13s
- `pytest -v tests/test_configurator.py`: 11/11 PASSED in 1.30s
- **Full Test Suite (`pytest`):** 146/146 PASSED in 25.15s (100% pass rate, zero regressions)

---

## Orchestrator Handoff Context
- Subtasks 1 and 2 of Phase 9 are 100% complete and fully verified:
  - Global benchmark de-biasing active: Non-coder models strictly bounded to SWE <= 25.0% and HumanEval <= 65.0% during imputation.
  - Domain classification & affinity bonuses verified across all 6 domains (+0.10 domain bonus, +0.15 Colibri AVX-512 CPU bonus).
  - Backend-aware TargetTokSec operational across CPU (8.0/15.0/5.0) and GPU (15.0/30.0/8.0).
  - Runtime environment configurator injector updated to emit `AGENT_COMPANION_BACKEND`, `AGENT_COMPANION_CONTEXT_LENGTH` (4096 / 8192 / 32768), `AGENT_COMPANION_MAX_RAM_MB=5200.0`, `AGENT_COMPANION_RAM_CEILING_MB=6144.0`, and `AGENT_COMPANION_AIRGAP_ENFORCED=true`.
  - Python runtime launcher template enforces `HardwareProfiler.RAM_CEILING_MB <= 6144.0` assertion on CPU backends.
  - Zero regressions across the entire 146-test suite.

---

## Active Gotchas & Failed Paths
- Colibri 1.49 bits rate on 27.0B parameters yields `(27e9 * 1.49) / (8 * 1024^2) = 4795.7897 MB` (~4795.79 MB), satisfying `<= 4800.0 MB`.
- In `test_context_clamping_27b_and_7b_cpu`, Qwen 7B at 8192 context has initial FP16 KV cache of 448 MB (total 5328 MB > 0.80 * 6144 MB = 4915.2 MB), which automatically and correctly triggers Q8_0 KV cache fallback to 224 MB (total 5104 MB <= 6144 MB).
- Under Phase 9 max_iq weighting profile (0.70/0.15/0.15), total score for top models is bounded by `0.70 * 1.0 + 0.15 * 1.0 = 0.8500`. Tied candidate models resolve via parameter count descending.

---

## Next Immediate Action
- Hand off completed Subtask 2 results to Orchestrator agent via `send_message`.
