"""Hardware profiling, GPU detection, and host capability subsystem."""

from companion_packager.hardware.detector import (
    check_disk_space_preflight,
    detect_hardware,
    get_memory_bandwidth_gb_s,
)
from companion_packager.hardware.gpu_detector import detect_gpus
from companion_packager.schemas import GPUInfo, HardwareProfile

__all__ = [
    "detect_hardware",
    "get_memory_bandwidth_gb_s",
    "check_disk_space_preflight",
    "detect_gpus",
    "GPUInfo",
    "HardwareProfile",
]
