"""Compatibility module re-exporting from companion_packager.schemas."""

from companion_packager.schemas import (
    CandidateModel,
    GPUInfo,
    HardwareProfile,
    Scorecard,
    SizingEvaluation,
    UserNeeds,
)

__all__ = [
    "GPUInfo",
    "HardwareProfile",
    "UserNeeds",
    "CandidateModel",
    "SizingEvaluation",
    "Scorecard",
]
