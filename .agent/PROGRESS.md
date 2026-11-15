# Agent Companion Packager - Progress & Tracking

## Current Active Phase & Status
- **Current Phase:** Phase 2 Complete: Hardware & Device Profiling Subsystem
- **Status:** PASSED. Zero-PyTorch hardware profiler, multi-platform GPU detector (NVIDIA, Apple Silicon, Windows CIM+Registry, Linux DRM), memory bandwidth heuristics, and disk space preflight implemented and verified with 100% test pass rate (46/46 tests passing across entire test suite).
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
   - `detect_hardware()`: Integrates `psutil` virtual memory, CPU core counts, `cpuinfo` SIMD flags (`has_avx512` checking `avx512vnni`, `avx512_vnni`, and `avx512f`; `has_avx2`; `has_neon`), zero-PyTorch GPU probing, and root/target disk space.
   - Backend target selector: Deterministically selects `cuda` (NVIDIA dGPU >= 4GB), `metal` (Apple Silicon), `rocm` (AMD dGPU >= 4GB), or `cpu` fallback.
   - `get_memory_bandwidth_gb_s()`: Deterministic sustained bandwidth mapping for discrete GPUs (RTX 4090/3090 -> 1000 GB/s, 4080/3080 -> 750 GB/s, 4070/3070 -> 500 GB/s, 4060/3060 -> 300 GB/s, fallback -> 250 GB/s), Apple Silicon tiers (Ultra -> 800 GB/s, Max -> 350 GB/s, Pro -> 175 GB/s, Base -> 100 GB/s), and CPU architectures (DDR5/12-14th Gen/Zen 4-5 -> 50 GB/s, DDR4/10-11th Gen/Zen 2-3 -> 30 GB/s, legacy -> 25 GB/s).
   - `check_disk_space_preflight()`: Strict ratio validation for build (2.5x model weights) vs download (1.5x model weights).

---

## Files Created, Modified, or Removed
- `src/companion_packager/hardware/__init__.py`: Hardware module package init exporting public API.
- `src/companion_packager/hardware/detector.py`: Core hardware profiler, bandwidth heuristics, and disk preflight.
- `src/companion_packager/hardware/gpu_detector.py`: Multi-platform zero-PyTorch GPU detection subsystem.
- `tests/test_hardware.py`: 13 comprehensive unit tests covering live detection, nvidia-smi parsing, registry qwMemorySize (int and little-endian binary), Apple Silicon, bandwidth heuristics, disk preflight, and backend target selection.
- `tests/test_hardware_detector.py`: Plan-compatible re-export test file.
- `.agent/PROGRESS.md`: Updated with Phase 2 status and test metrics.

---

## Test Results & Verification Logs
- `pytest -v tests/test_hardware.py`:
  - `test_live_detect_hardware`: PASSED (live host probed: 11th Gen Core i7, 16GB RAM, AVX-512, Iris Xe GPU, 30 GB/s DDR4 target, backend "cpu").
  - `test_parse_nvidia_smi_single_gpu`: PASSED (RTX 4090 24GB parsed).
  - `test_parse_nvidia_smi_multi_gpu`: PASSED (RTX 3060 + RTX 4090 parsed, max VRAM selected).
  - `test_parse_nvidia_smi_malformed_lines`: PASSED (gracefully handles malformed CSV lines).
  - `test_detect_nvidia_gpus_subprocess_mock`: PASSED (CLI subprocess mocking and FileNotFoundError handling).
  - `test_parse_registry_qwmemory_int_and_bytes`: PASSED (integers and 8-byte little-endian binary payloads parsed).
  - `test_windows_gpus_cim_and_registry_mock`: PASSED (PowerShell CIM + registry 12GB VRAM override verified).
  - `test_apple_silicon_detection_mock`: PASSED (Apple M2 Max with 32GB unified memory and Metal driver verified).
  - `test_bandwidth_heuristics_gpu`: PASSED (4090/3090, 4080/3080, 4070/3070, 4060/3060, AMD discrete GPU).
  - `test_bandwidth_heuristics_apple_silicon`: PASSED (Ultra, Max, Pro, Base).
  - `test_bandwidth_heuristics_cpu_tiers`: PASSED (DDR5/13th Gen/Zen 4 -> 50.0, DDR4/11th Gen/Zen 3 -> 30.0, Legacy -> 25.0).
  - `test_check_disk_space_preflight_build_and_download`: PASSED (2.5x build vs 1.5x download constraints).
  - `test_detect_hardware_backend_target_selection`: PASSED (cuda, metal, rocm, and cpu fallback verified).
- Full Test Suite: `pytest -v tests/` -> 46/46 passed in 4.79s.
- `python -m build --wheel`: `companion_packager-0.1.0-py3-none-any.whl` built cleanly.

---

## Orchestrator Handoff Context
- Phase 2 complete. Hardware profiling subsystem operational and thoroughly tested across all platforms and mock targets.
- Host machine correctly identified as 11th Gen Intel i7-1165G7 with AVX-512 support, Iris Xe GPU, and 30 GB/s DDR4 bandwidth.
- Zero external PyTorch/CUDA dependencies introduced.
- Ready for Phase 3: Research Subsystem & Benchmarks (`research/hf_client.py`, `research/benchmarks.py`, `research/benchmarks_cache.json`, `research/citations.py`).

---

## Active Gotchas & Failed Paths
- None. Windows `qwMemorySize` parsing defensively accommodates both `int` and `REG_BINARY` byte buffers across diverse vendor display drivers without runtime exceptions.

---

## Next Immediate Action
- Hand off to Orchestrator for Phase 3 dispatch (Model Research, Hugging Face Hub metadata querying, offline benchmark dataset, and citation generator).
