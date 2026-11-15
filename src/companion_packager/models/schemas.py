"""Compatibility module re-exporting from companion_packager.schemas."""

from companion_packager.schemas import (
    CandidateModel,
    GPUInfo,
    HardwareProfile,
    ProofCitation,
    Scorecard,
    SizingEvaluation,
    UserNeeds,
)

__all__ = [
    "GPUInfo",
    "HardwareProfile",
    "UserNeeds",
    "ProofCitation",
    "CandidateModel",
    "SizingEvaluation",
    "Scorecard",
]
