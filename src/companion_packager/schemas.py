"""Pydantic v2 domain schemas for companion-packager."""

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class GPUInfo(BaseModel):
    """GPU hardware device metadata."""
    index: int
    name: str
    vendor: str = Field(description="GPU vendor: nvidia, amd, apple, or intel")
    vram_total_mb: float
    vram_free_mb: float = 0.0
    cuda_compute_capability: str = ""
    driver_version: str = ""


class HardwareProfile(BaseModel):
    """Host machine hardware specification and runtime capabilities."""
    os_name: str
    cpu_brand: str
    cpu_arch: str
    physical_cores: int
    logical_cores: int
    has_avx512: bool
    has_avx2: bool
    has_neon: bool
    ram_total_mb: float
    ram_avail_mb: float
    disk_free_gb: float
    gpus: List[GPUInfo] = Field(default_factory=list)
    bandwidth_gb_s: float = 30.0
    backend_target: str = "cpu"


class UserNeeds(BaseModel):
    """User operational constraints and workload preferences."""
    task: str = Field(
        default="coding",
        description="Workload domain: coding, creative_writing, general_desktop, reasoning_math, summarization, custom",
    )
    priority: str = Field(default="balanced", description="Optimization goal: balanced, speed, or max_iq")
    min_context_tokens: int = Field(default=16384, description="Minimum required context length in tokens")
    require_permissive_license: bool = Field(default=False, description="Strictly filter for Apache-2.0 or MIT")
    custom_intent: str = Field(default="", description="Freeform user intent description when task=custom")


class ProofCitation(BaseModel):
    """Empirical 3-pillar proof citation for a model recommendation."""
    model_id: str
    task_benchmark_name: str
    task_benchmark_score: float
    is_measured: bool = False
    ram_allocated_mb: float
    ram_headroom_mb: float
    tokens_per_second: float
    justification_bullets: List[str] = Field(default_factory=list)


class CandidateModel(BaseModel):
    """Model checkpoint metadata, architecture parameters, and benchmark scores."""
    repo_id: str
    filename: str
    quant_method: str
    param_count_billions: float
    active_weight_ratio: float = 1.0
    size_mb: float
    context_window: int = 32768
    arch_type: str = "dense"
    num_layers: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    mla_v_head_dim: int = 128
    mla_kv_lora_rank: int = 512
    mla_rope_head_dim: int = 64
    benchmarks: Dict[str, float] = Field(default_factory=dict)
    domain_benchmarks: Dict[str, float] = Field(default_factory=dict)
    measured_benchmarks: List[str] = Field(default_factory=list)
    proof_justification: str = ""
    citation_doi: str = ""
    paper_url: str = ""
    license: str = "apache-2.0"


class SizingEvaluation(BaseModel):
    """Hardware feasibility, memory budget, and projected throughput evaluation."""
    fits: bool
    weights_mb: float
    kv_cache_mb: float
    overhead_mb: float
    total_ram_mb: float
    budget_ceiling_mb: float
    tokens_per_second: float
    ram_penalty: float = 0.0
    rejection_reason: str = ""

    @property
    def weights_ram_mb(self) -> float:
        return self.weights_mb

    @property
    def runtime_ram_mb(self) -> float:
        return self.total_ram_mb


class Scorecard(BaseModel):
    """Scored and ranked candidate model with hardware sizing evaluation."""
    candidate: CandidateModel
    evaluation: SizingEvaluation
    task_fit_score: float
    latency_score: float
    final_score: float
    rank: int = 1
    proof_citation: Optional[ProofCitation] = None


__all__ = [
    "GPUInfo",
    "HardwareProfile",
    "UserNeeds",
    "ProofCitation",
    "CandidateModel",
    "SizingEvaluation",
    "Scorecard",
]

