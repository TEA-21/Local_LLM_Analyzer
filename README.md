# Local LLM Analyzer & Packager (`companion-packager`)

An autonomous, hardware-aware CLI tool, model researcher, citation generator, and installer packager for local LLMs and **Agent Companion**.

Profiles target PC hardware (CPU SIMD vector extensions, physical unswapped RAM, discrete and unified GPUs, memory bus bandwidth), interactively interviews user workload requirements, discovers and ranks models from Hugging Face Hub, outputs empirical **3-Pillar Proof Citations**, and builds production installers (Windows Inno Setup `.exe`, Linux/macOS POSIX `.run` SFX archives, or portable bundles).

---

## Key Capabilities

### 1. Zero-PyTorch Hardware & Memory Bus Profiler
- **CPU SIMD Vector Inspection:** Probes native instruction flags (`avx512vnni`, `avx512_vnni`, `avx512f`, `avx2`, `fma`, `neon`) to unlock hardware-accelerated kernels.
- **Zero-PyTorch GPU Probing:** Completely decoupled from PyTorch/CUDA runtimes for instant sub-second startup:
  - NVIDIA discrete GPUs queried directly via `nvidia-smi` CSV queries.
  - Apple Silicon unified memory queried via `sysctl` and macOS system profiler.
  - Windows dGPU VRAM queried via PowerShell CIM with safe 64-bit registry fallback (`HardwareInformation.qwMemorySize` supporting both integer and 8-byte little-endian binary types).
  - Linux DRM sysfs probing (`/sys/class/drm/card*/device/mem_info_vram_total`).
- **Memory Bandwidth Modeling:** Maps sustained memory bus bandwidth (GB/s) to model autoregressive token generation latency ($B_{\text{sustained}} / M_{\text{weights}}$).
- **Clean-State Host RAM Ceiling:** Enforces Agent Companion's clean-state memory invariant:
  $$M_{\text{budget\_ceiling}} = \min(M_{\text{ram\_total}} \times 0.70, 6144.0\text{ MB})$$
  guaranteeing unswapped execution on 8GB and 16GB host machines without triggering swapfile thrashing.

### 2. Multi-Domain Workload Discovery & Intent Planner
Translates user requirements into targeted model discovery across 6 core domains without requiring manual model names:
- **Coding & AST Agency:** Software engineering, Hermes tooling, multi-file synthesis (`SWE-bench Verified`, `HumanEval`).
- **Creative Writing & Prose:** Character dialogue, world-building, narrative voice (`EQ-Bench`, `MT-Bench Writing/Humanities`).
- **General Desktop Assistant:** Fast instruction following, system control (`IFEval`, `Chatbot Arena Elo`).
- **Reasoning & Formal Math:** Step-by-step chain-of-thought derivations (`GSM8K`, `MATH`, `OpenLLM v2`).
- **Summarization:** Long-context synthesis, document distillation (`MMLU-Pro`, `Chatbot Arena Elo`).
- **Custom Open-Ended Intent:** Natural language query planner that extracts domain tokens, strips stopwords, and autonomously queries Hugging Face Hub.

### 3. Empirical 3-Pillar "Proof Citation" Engine
Replaces dry academic paper DOIs with verifiable proof cards explaining exactly why a model was chosen:
- **Pillar 1: Task-Specific Benchmark Proof:** Directly measured leaderboard metrics (`[Measured]`) or calibrated two-way cross-imputations (`[Estimated]`) with domain de-biasing guards.
- **Pillar 2: Hardware Fit & Headroom Proof:** Exact memory allocation ($M_{\text{weights}} + M_{\text{kv\_cache}} + M_{\text{runtime\_overhead}}$), safety margin under the budget ceiling, and calculated generation speed (tok/s).
- **Pillar 3: Plain-Language Justification:** Concise bullets explaining domain fit for the user's workload.

### 4. Agent Companion Engine Harmonization (Colibri & SpikeBrain)
- **Colibri Sub-2-bit (Ternary 1.49b) Integration:**
  - Evaluates models with ternary quantization (`candidate.quant_method == "TERNARY"` or `ternaryquench`):
    $$M_{\text{weights}} = \frac{27 \times 10^9 \times 1.49}{8 \times 1024^2} = 4,795.8\text{ MB} \le 4,800.0\text{ MB}$$
  - Runtime overhead strictly $400.0\text{ MB}$, fitting a 27B model into a sub-5.2GB envelope.
- **SpikeBrain 15% Neuromorphic Sparsity on AVX-512:**
  - Dynamic neuromorphic weight gating schedules 15% active weights per token ($S = 0.60$ bandwidth boost).
  - Achieves $10.0\text{--}10.2\text{ tok/s}$ sustained throughput on standard 30 GB/s DDR4 memory buses.
- **CPU Context Clamping:**
  - Automatically clamps active context tokens on CPU hosts to prevent KV cache memory blowout:
    - 27B models clamped to $C \le 4,096$ tokens (Q8_0 KV cache = $512.0\text{ MB}$, total $M_{\text{runtime}} = 5,707.8\text{ MB} \le 6,144.0\text{ MB}$).
    - $\ge 7\text{B}$ models clamped to $C \le 8,192$ tokens (KV cache $\le 512.0\text{ MB}$).

### 5. Resumable Chunked Downloader
- HTTP streaming via `httpx` with `follow_redirects=True`.
- Handles HTTP 206 (range append to `.part`), HTTP 200 (truncate and stream), and HTTP 416 (length match verification).
- Idempotency check skips existing downloads if full-file SHA-256 matches.
- Defends against weak ETags (`W/`), multipart suffixes (`-2`), and verifies LFS checksums.

### 6. Cross-Platform Distributable Packaging Engine
- **Thin Web-Installer:** Lightweight bundle that profiles the target PC, downloads the optimal model on-site, configures environment invariants, and installs Agent Companion.
- **Offline Fat-Installer:** Self-contained bundle containing vendorized wheels, standalone Python runtime (`install_only.tar.gz`), pre-downloaded model weights, and offline configurations.
- **Inno Setup Builder:** Compiles native Windows `AgentCompanionSetup.exe` with `#define SourceDir`, non-admin `{localappdata}` defaults, or falls back to portable execution bundles.
- **POSIX SFX Archive:** Compiles Linux/macOS `.run` self-extracting archive with bash extraction headers and desktop shortcuts.

---

## Recommendation Matrix Across Hardware Tiers

| Hardware Tier | Workload | Top Recommended Model | Score | RAM / VRAM | Speed |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Local PC (AVX-512, 16GB RAM)** | Coding | `penkia/TernaryQuench-Qwen3.8-27B-GGUF` | **0.850** | 5,707.8 MB | 10.2 tok/s |
| | Creative Writing | `bartowski/Mistral-7B-Instruct-v0.3-GGUF` | **0.719** | 5,082.0 MB | 7.0 tok/s |
| | General Desktop | `penkia/TernaryQuench-Qwen3.8-27B-GGUF` | **0.800** | 5,707.8 MB | 10.2 tok/s |
| | Reasoning / Math | `unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF` | **0.777** | 1,453.0 MB | 38.2 tok/s |
| | Summarization | `Qwen/Qwen2.5-7B-Instruct-GGUF` | **0.652** | 5,104.0 MB | 6.6 tok/s |
| **8GB Laptop (No AVX-512, DDR4)** | Coding | `penkia/TernaryQuench-Qwen3.8-27B-GGUF` | **0.780** | 5,707.8 MB | 6.4 tok/s |
| | Creative Writing | `bartowski/Mistral-7B-Instruct-v0.3-GGUF` | **0.706** | 5,082.0 MB | 7.0 tok/s |
| | General Desktop | `bartowski/Llama-3.2-3B-Instruct-GGUF` | **0.740** | 3,146.0 MB | 15.0 tok/s |
| | Reasoning / Math | `unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF` | **0.777** | 1,453.0 MB | 38.2 tok/s |
| | Summarization | `Qwen/Qwen2.5-3B-Instruct-GGUF` | **0.635** | 2,826.0 MB | 15.0 tok/s |
| **RTX 4060 (8GB VRAM)** | Coding | `Qwen/Qwen2.5-Coder-3B-Instruct-GGUF` | **0.735** | 3,126.0 MB | 15.0 tok/s |
| | General Desktop | `bartowski/Llama-3.2-3B-Instruct-GGUF` | **0.740** | 3,446.0 MB | 15.0 tok/s |
| | Reasoning / Math | `unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF` | **0.777** | 1,753.0 MB | 38.2 tok/s |
| **RTX 4090 (24GB VRAM)** | Coding | `bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF` | **0.841** | 10,586.0 MB | 21.3 tok/s |
| | Reasoning / Math | `unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF` | **0.777** | 1,753.0 MB | 38.2 tok/s |

---

## Installation

### Prerequisites
- Python >= 3.11
- `uv` (recommended) or `pip`

```bash
# Clone repository
git clone https://github.com/TEA-21/Local_LLM_Analyzer.git
cd Local_LLM_Analyzer

# Install dependencies and package in editable mode
uv pip install -e .
```

---

## CLI Usage

### 1. Hardware Profile Scan
Profiles CPU SIMD extensions, physical and available RAM, discrete GPUs, VRAM, and memory bus bandwidth:
```bash
companion-packager profile
companion-packager profile --json
```

### 2. Interactive or Headless Model Recommendation
Interviews user requirements and outputs empirical 3-pillar proof scorecards:
```bash
# Interactive mode
companion-packager recommend

# Headless mode across domains
companion-packager recommend --task coding --priority balanced
companion-packager recommend --task creative_writing --priority quality
companion-packager recommend --task general_desktop --priority speed
companion-packager recommend --task reasoning_math --context-len 8192

# Custom natural language intent
companion-packager recommend --task custom --custom-intent "financial quarterly report analysis and table extraction"

# 100% offline airgapped recommendation
companion-packager recommend --task coding --offline
```

### 3. Resumable Model Checkpoint Download
Downloads GGUF model weights with resume capability, progress bars, and SHA-256 verification:
```bash
companion-packager download penkia/TernaryQuench-Qwen3.8-27B-GGUF TernaryQuench-Qwen3.8-27B-Q2_K.gguf --output-dir models/
```

### 4. Build Distributable Installer
Builds a Thin (online download) or Fat (offline airgapped) installer:
```bash
# Windows Inno Setup executable
companion-packager build --mode thin --format inno

# Offline airgapped standalone bundle
companion-packager build --mode fat --format portable --companion-path "D:/Projects/Agent Companion"
```

### 5. Target Machine Installation
Executed on target machines by the installer runner:
```bash
companion-packager install --install-dir "$HOME/.local/share/agent-companion" --non-interactive
```

---

## Architecture & Code Structure

```
Local_LLM_Analyzer/
├── pyproject.toml                  # PEP 621 packaging (hatchling backend)
├── README.md                       # Comprehensive documentation
├── src/
│   └── companion_packager/
│       ├── __init__.py
│       ├── __main__.py             # python -m companion_packager entrypoint
│       ├── cli.py                  # Typer CLI application
│       ├── schemas.py              # Pydantic v2 domain models
│       ├── hardware/
│       │   ├── detector.py         # CPU SIMD, RAM, memory bandwidth heuristics
│       │   └── gpu_detector.py     # Zero-PyTorch GPU detection (CUDA, Metal, ROCm)
│       ├── interview/
│       │   └── questionnaire.py    # Multi-domain questionnaire & intent tokenizer
│       ├── research/
│       │   ├── hf_client.py        # Hugging Face API client & query planner
│       │   ├── benchmarks.py       # Bidirectional cross-imputation & normalization
│       │   ├── benchmarks_cache.json # Airgapped stratified benchmark dataset
│       │   └── citations.py        # 3-Pillar Proof Citation generator
│       ├── sizing/
│       │   ├── formulas.py         # MHA/GQA/MLA KV cache, Colibri 1.49b, SpikeBrain
│       │   └── recommender.py      # Scoring matrix, domain affinity, TargetTokSec
│       └── installer/
│           ├── downloader.py       # Resumable chunked downloader (206/200/416)
│           ├── configurator.py     # Agent Companion environment & launcher injector
│           ├── wheel_vendor.py     # Third-party PyPI wheel compilation & ABI match
│           ├── inno_builder.py     # Inno Setup script compiler & portable fallback
│           └── sfx_builder.py      # POSIX shell self-extracting archive packager
└── tests/                          # 146 unit & integration tests
```

---

## Verification & Test Suite

The test suite contains 146 automated tests covering mathematical hardware sizing, cross-imputation graphs, GPU detection, downloader resumption, and packaging dry runs:

```bash
uv run pytest -v
```

```
============================== 146 passed in 21.94s ==============================
```

---

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
