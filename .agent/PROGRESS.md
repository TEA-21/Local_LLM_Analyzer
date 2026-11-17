# Agent Companion Packager - Progress & Tracking

## Current Active Phase & Status
- **Current Phase:** Phase 10: Dynamic Process Audit, 3-Scenario Operational Concurrency & Multi-Tier Model Optimization
- **Subtask Status:** Phase 10 Subtask 4 (Concurrency Recommender & Interactive Table CLI) 100% COMPLETE & VERIFIED. Phase 10 fully complete.
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
11. **Phase 10 Subtask 1 - Process Inspection Subsystem & Domain Schemas:**
    - Defined `ProcessCategory(str, Enum)` with categories: `browsers`, `ides`, `dev_runtimes`, `system`, `other`.
    - Defined `ProcessMemorySummary(BaseModel)` with category megabytes, total scanned MB, and top 5 process list.
    - Defined `ConcurrencyScenario(str, Enum)` with `parallel`, `minimal_interference`, and `dedicated`.
    - Defined `ConcurrencyScenarioBudget(BaseModel)` with scenario, title, description, RAM stats, budget ceiling, and recommended scorecard.
    - Updated `src/companion_packager/schemas.py` and converted `src/companion_packager/models/schemas.py` into a re-export shim (`from companion_packager.schemas import *`).
    - Implemented `audit_active_processes() -> ProcessMemorySummary` in `src/companion_packager/hardware/process_inspector.py`:
      - Cross-platform executable normalization (`removesuffix('.exe')`).
      - Delimited token-bounded matching to eliminate false positive collisions (e.g., `codec`, `mongodb`, `xcodebuild` correctly mapped to `other`).
      - Robust error handling silently catching `NoSuchProcess`, `AccessDenied`, and `ZombieProcess`.
      - Extracted top 5 processes by RSS descending with name, rss_mb, category, and pid.
    - Re-exported `audit_active_processes`, `ProcessCategory`, and `ProcessMemorySummary` in `src/companion_packager/hardware/__init__.py`.
    - Created unit & live integration tests in `tests/test_process_inspector.py`.
12. **Phase 10 Subtask 2 - Concurrency Scenario Budget Formulation & Sizing Overrides (`formulas.py`, `sizing/__init__.py`):**
    - Updated `evaluate_sizing()` signature with `budget_ceiling_override_mb: Optional[float] = None` and default `context_tokens: int = 16384`.
    - Implemented override bypass: `budget_ceiling_override_mb` takes immediate precedence over `calculate_budget_ceiling_mb(hardware, clean_state=clean_state)`.
    - Implemented `calculate_scenario_budgets(hardware, process_summary=None) -> Dict[ConcurrencyScenario, ConcurrencyScenarioBudget]`:
      - Automatic fallback to `audit_active_processes()` when `process_summary is None`.
      - CPU Host Branch:
        - $M_{\text{ceiling\_host}} = \min(M_{\text{ram\_total}} \times 0.70, 6144.0\text{ MB})$.
        - Non-reclaimable memory: $M_{\text{ides}} + M_{\text{dev\_runtimes}} + M_{\text{system}}$.
        - Parallel: $M_{\text{budget\_parallel}} = \min(M_{\text{ram\_avail}} \times 0.75, M_{\text{ceiling\_host}})$, $M_{\text{reclaimable}} = 0.0\text{ MB}$.
        - Minimal Interference: $M_{\text{reclaimable}} = 0.50 \times (M_{\text{browsers}} + M_{\text{other}})$, effective available memory clamped by $\max(1024.0, M_{\text{ram\_total}} - \text{non\_reclaimable})$, $M_{\text{budget\_minimal}} = \min(\text{effective\_available} \times 0.80, M_{\text{ceiling\_host}})$.
        - Dedicated Alone: $M_{\text{reclaimable}} = M_{\text{browsers}} + M_{\text{other}}$, $M_{\text{budget\_dedicated}} = \min(M_{\text{ceiling\_host}}, \max(M_{\text{ram\_avail}}, M_{\text{ram\_total}} - \text{non\_reclaimable}) \times 0.85)$.
      - GPU Execution Branch:
        - Free VRAM detection with fallback to $0.80 \times M_{\text{vram\_total}}$ when free VRAM is missing/0.
        - Host ceiling: $\min(M_{\text{vram\_total}} \times 0.80, M_{\text{vram\_free}} \times 0.95)$.
        - Parallel budget: $\min(M_{\text{vram\_free}} \times 0.80, M_{\text{ceiling\_host}})$.
        - Minimal budget: $\min(M_{\text{vram\_free}} \times 0.90, M_{\text{ceiling\_host}})$.
        - Dedicated budget: $M_{\text{ceiling\_host}}$.
      - Guarded against empty GPU lists by falling back safely to CPU host formulations.
    - Exported `calculate_scenario_budgets` in `src/companion_packager/sizing/__init__.py`.
    - Added unit and scenario tests covering CPU, GPU, override precedence, reclaimable RAM clamping under heavy Docker/WSL2 holds, and empty GPU fallback in `tests/test_formulas.py` and `tests/test_sizing_formulas.py`.
13. **Phase 10 Subtask 3 - Airgapped Benchmarks Cache Expansion & Multi-Tier Model Optimization (`benchmarks_cache.json`, `benchmarks.py`, `citations.py`):**
    - Expanded `src/companion_packager/research/benchmarks_cache.json` from 15 to 20 models by appending 5 optimized candidate models:
      - `bartowski/Meta-Llama-3.1-8B-Instruct-GGUF` at `Q3_K_M` (3740.0 MB, EQ-Bench 74.8, MT-Bench 8.6, Arena-Elo 1200.0, 32 layers, 8 KV heads, 128 head_dim).
      - `Qwen/Qwen2.5-7B-Instruct-GGUF` at `Q3_K_M` (3560.0 MB, IFEval 80.4, MMLU-Pro 56.2, Arena-Elo 1205.0, 28 layers, 4 KV heads, 128 head_dim).
      - `Qwen/Qwen2.5-14B-Instruct-GGUF` at `Q3_K_M` (6980.0 MB, IFEval 85.1, MMLU-Pro 64.8, Arena-Elo 1260.0, 48 layers, 8 KV heads, 128 head_dim).
      - `unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF` at `Q3_K_M` (3560.0 MB, GSM8K 91.4, MATH 77.5, 28 layers, 4 KV heads, 128 head_dim).
      - `unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF` at `Q3_K_M` (6980.0 MB, GSM8K 93.9, MATH 80.2, 48 layers, 8 KV heads, 128 head_dim).
    - Implemented `resolve_custom_domain(custom_intent: Optional[str]) -> str` in `src/companion_packager/research/benchmarks.py` with multi-token keyword resolution across `coding`, `reasoning_math`, `creative_writing`, `summarization`, and `general_desktop`.
    - Added `custom_intent: Optional[str] = None` to `normalize_benchmarks`, dynamically routing `task="custom"` through `resolve_custom_domain`.
    - Updated `generate_proof_citation` in `src/companion_packager/research/citations.py` to route `task="custom"` using `resolve_custom_domain(custom_intent)` for Pillar 1 benchmark selection, and verified provenance tags (`[Measured]` vs `[Estimated]`).
    - Connected `custom_intent=user_needs.custom_intent` through `recommender.py` and `generate_recommendations_markdown`.
    - Re-exported `resolve_custom_domain` via `companion_packager.research`.
14. **Phase 10 Subtask 4 - Concurrency Recommender & Interactive Table CLI (`recommender.py`, `cli.py`, `configurator.py`, `schemas.py`):**
    - Updated `rank_candidates(...)` in `src/companion_packager/sizing/recommender.py`:
      - Added `budget_ceiling_override_mb: Optional[float] = None` and forwarded directly to `evaluate_sizing`.
      - Refined domain classification rules: `is_code_model`, `is_prose_model` (`llama-3.1`/`mistral` excluding `3.2-3b`), `is_summary_model` (`qwen2.5` non-coder), `is_math_model` (`r1`/`math`), and `is_desktop_model` (including `qwen2.5-7b` under `general_desktop`/`custom`).
    - Implemented `evaluate_all_concurrency_scenarios(...)`:
      - Automatically executes process memory audit or takes provided `ProcessMemorySummary`.
      - Generates scenario budgets for `PARALLEL`, `MINIMAL_INTERFERENCE`, and `DEDICATED`.
      - Evaluates and ranks candidate models per scenario using `hard_exclude=True` and scenario budget ceiling overrides.
      - Populates `recommended_scorecard` with top valid candidate that fits.
    - Implemented `display_concurrency_scenarios_table(...)`:
      - Displays active process breakdown across Browsers, IDEs, Dev Runtimes, Other, and Scanned Total.
      - Formats Rich terminal table displaying Scenario, RAM Budget, Recommended Model, Quant, Est. Speed, RAM Footprint, Headroom, and Headline Proof / Benchmark metrics.
      - Gracefully handles scenarios where no candidates fit with red warning status.
    - Exported `evaluate_all_concurrency_scenarios` and `display_concurrency_scenarios_table` in `src/companion_packager/sizing/__init__.py`.
    - Extended `src/companion_packager/schemas.py`:
      - Added `headline_metric_name`, `headline_metric_value`, and `headline_metric_tag` properties on `ProofCitation`.
    - Updated `src/companion_packager/cli.py`:
      - `recommend` command: Added `--scenario` option, dynamically audits processes, displays 3-scenario table, prompts user in interactive mode (or applies CLI flag / defaults to parallel), renders full scorecard and proof citation for chosen scenario, and exports `RECOMMENDATIONS.md`.
      - `build` command: Added `--task`, `--priority`, `--scenario`, `--target-os`, `--target-arch` options. Computes scenario budget, injects scenario config, and writes `installer_manifest.json` into bundle directory.
      - `install` command: Added `--manifest` and `--scenario` options. Resolves manifest from bundle or install directory, extracting scenario and model path for offline installation.
    - Updated `src/companion_packager/installer/configurator.py`:
      - Added `scenario: Optional[ConcurrencyScenarioBudget] = None` to `inject_agent_companion_config` and aliased `generate_config`.
      - Sets `AGENT_COMPANION_MAX_RAM_MB = round(scenario.budget_ceiling_mb, 1)` when scenario is provided.
      - Enforces `AGENT_COMPANION_RAM_CEILING_MB = 6144.0` CPU invariant.
      - Injected runtime warning into `start_companion.py` if free RAM drops below 70% of scenario threshold.
    - Exported `generate_config` in `src/companion_packager/installer/__init__.py`.

---

## Files Created, Modified, or Removed
- **Created:**
  - `src/companion_packager/hardware/process_inspector.py`
  - `tests/test_process_inspector.py`
- **Modified:**
  - `src/companion_packager/schemas.py`
  - `src/companion_packager/models/schemas.py`
  - `src/companion_packager/hardware/__init__.py`
  - `src/companion_packager/sizing/formulas.py`
  - `src/companion_packager/sizing/__init__.py`
  - `src/companion_packager/sizing/recommender.py`
  - `src/companion_packager/research/__init__.py`
  - `src/companion_packager/research/benchmarks.py`
  - `src/companion_packager/research/benchmarks_cache.json`
  - `src/companion_packager/research/citations.py`
  - `src/companion_packager/installer/configurator.py`
  - `src/companion_packager/installer/__init__.py`
  - `src/companion_packager/cli.py`
  - `tests/test_sizing_formulas.py`
  - `tests/test_formulas.py`
  - `tests/test_research.py`
  - `tests/test_domain_proof.py`
  - `tests/test_recommender.py`
  - `tests/test_configurator.py`
  - `tests/test_cli.py`

---

## Test Results & Verification Logs
- `pytest -v tests/test_recommender.py tests/test_cli.py tests/test_configurator.py`: 43/43 PASSED in 31.27s (100% pass rate)
- Full project test suite `pytest`: 176/176 PASSED in 63.53s (100% pass rate, zero regressions)

---

## Orchestrator Handoff Context
- Phase 10 Subtask 4 and the entirety of Phase 10 are 100% complete and verified:
  - 3-scenario operational concurrency evaluation and terminal reporting are active across `evaluate_all_concurrency_scenarios`, `display_concurrency_scenarios_table`, and `companion-packager recommend`.
  - On 16GB CPU host, in Dedicated mode: 7B/8B/Colibri models win across Creative Writing, Summarization, and Coding.
  - In tight Parallel mode: 1.5B/3B models win to avoid host multitasking interference.
  - `build` stages `installer_manifest.json` and injects exact scenario RAM thresholds.
  - `install` auto-detects manifest configuration for zero-friction setup.
  - Full test suite passes with 176 passing tests.

---

## Active Gotchas & Failed Paths
- Substring checks on process names cause collisions (e.g., `code` in `codec`, `go` in `mongodb`); exact token matching with delimiter substitution (`-`, `_`) strictly eliminates false positives.
- `psutil.process_iter` can throw transient `NoSuchProcess` or `AccessDenied` on ephemeral host processes; caught and ignored silently.
- When `primary_gpu.vram_free_mb` is 0.0 or unqueried, GPU ceiling must fall back to `primary_gpu.vram_total_mb * 0.80` instead of multiplying `0.0 * 0.95`.
- Adding lighter Q3_K_M quant models can yield higher throughput and rank above Q4_K_M in balanced scorecards; test assertions filtering by repo_id must distinguish between quant methods when checking specific target fit scores.
- Terminal table cell wrapping splits multiline strings across lines in Rich console; tests asserting on table output should check tokens independently or account for column boundary splits.
- Standard library `json` must be explicitly imported in `cli.py` for manifest serialization and deserialization.

---

## Next Immediate Action
- Complete subagent handoff to Elephant Orchestrator. All Phase 10 tasks verified and closed.
