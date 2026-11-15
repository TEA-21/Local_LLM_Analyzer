"""Hardware profiling, SIMD feature detection, and memory bandwidth estimation.

Zero-PyTorch profiling leveraging psutil and py-cpuinfo for host capability
characterization and backend target recommendation.
"""

from pathlib import Path
import platform
import re
import sys
from typing import List, Optional, Tuple

import cpuinfo
import psutil

from companion_packager.hardware.gpu_detector import detect_gpus
from companion_packager.schemas import GPUInfo, HardwareProfile


def get_memory_bandwidth_gb_s(
    cpu_brand: str, arch: str, gpus: Optional[List[GPUInfo]] = None
) -> float:
    """Estimate sustained memory bandwidth in GB/s using deterministic hardware mapping.

    Hierarchy:
    1. Discrete NVIDIA / AMD GPU with >= 4GB VRAM
    2. Apple Silicon Unified Memory
    3. CPU RAM (DDR5 vs DDR4 vs Legacy)
    """
    gpus = gpus or []

    # 1. Check for discrete GPUs with significant VRAM
    discrete_gpus = [
        g for g in gpus if g.vendor in ("nvidia", "amd") and g.vram_total_mb >= 4000.0
    ]
    if discrete_gpus:
        # Select GPU with max VRAM
        primary_gpu = max(discrete_gpus, key=lambda g: g.vram_total_mb)
        name_upper = primary_gpu.name.upper()

        if "4090" in name_upper or "3090" in name_upper:
            return 1000.0
        if "4080" in name_upper or "3080" in name_upper:
            return 750.0
        if "4070" in name_upper or "3070" in name_upper:
            return 500.0
        if "4060" in name_upper or "3060" in name_upper:
            return 300.0
        return 250.0

    # 2. Check for Apple Silicon
    apple_gpus = [g for g in gpus if g.vendor == "apple"]
    is_apple_silicon = bool(apple_gpus) or (
        sys.platform == "darwin"
        and ("arm" in arch.lower() or "aarch64" in arch.lower() or "apple" in cpu_brand.lower())
    )

    if is_apple_silicon:
        combined_text = (
            cpu_brand.lower()
            + " "
            + " ".join(g.name.lower() for g in apple_gpus)
        )
        if "ultra" in combined_text:
            return 800.0
        if "max" in combined_text:
            return 350.0
        if "pro" in combined_text:
            return 175.0
        return 100.0

    # 3. CPU Memory Bandwidth mapping
    brand_lower = cpu_brand.lower()

    # Explicit DDR5 indicator
    if "ddr5" in brand_lower:
        return 50.0

    # Modern Intel 12th-14th Gen or Core Ultra
    # e.g. "12th Gen", "13th Gen", "14th Gen", "Core Ultra", "i7-12700", "i9-13900", "i5-14600"
    intel_modern_patterns = [
        r"1[234]th\s+gen",
        r"core\s+ultra",
        r"i[3579]-1[234]\d{3}",
    ]
    if any(re.search(p, brand_lower) for p in intel_modern_patterns):
        return 50.0

    # Modern AMD Zen 4 / Zen 5 (Ryzen 7000, 8000, 9000 series)
    amd_modern_patterns = [
        r"zen\s*[45]",
        r"ryzen\s+[3579]\s+[789]\d{3}",
    ]
    if any(re.search(p, brand_lower) for p in amd_modern_patterns):
        return 50.0

    # Explicit DDR4 indicator
    if "ddr4" in brand_lower:
        return 30.0

    # Intel 10th-11th Gen (e.g. Tiger Lake, Ice Lake, Comet Lake)
    intel_ddr4_patterns = [
        r"1[01]th\s+gen",
        r"i[3579]-1[01]\d{3}",
        r"tiger\s*lake",
        r"ice\s*lake",
        r"comet\s*lake",
    ]
    if any(re.search(p, brand_lower) for p in intel_ddr4_patterns):
        return 30.0

    # AMD Zen 2 / Zen 3 (Ryzen 3000, 4000, 5000 series)
    amd_ddr4_patterns = [
        r"zen\s*[23]",
        r"ryzen\s+[3579]\s+[345]\d{3}",
    ]
    if any(re.search(p, brand_lower) for p in amd_ddr4_patterns):
        return 30.0

    # Legacy or unclassified x86_64 fallback
    return 25.0


def detect_hardware(target_disk_path: Optional[Path] = None) -> HardwareProfile:
    """Probe host machine hardware specs and runtime capabilities.

    Probes physical & virtual RAM, CPU SIMD extensions, GPUs, disk space,
    and determines recommended backend target and sustained memory bandwidth.
    """
    os_name = platform.system()

    # Memory via psutil
    vm = psutil.virtual_memory()
    ram_total_mb = round(vm.total / (1024 * 1024), 2)
    ram_avail_mb = round(vm.available / (1024 * 1024), 2)

    # CPU Cores
    physical_cores = psutil.cpu_count(logical=False) or 1
    logical_cores = psutil.cpu_count(logical=True) or physical_cores

    # CPU Brand, Arch, and SIMD flags via cpuinfo
    cpu_info = cpuinfo.get_cpu_info()
    cpu_brand = (
        cpu_info.get("brand_raw")
        or cpu_info.get("brand")
        or platform.processor()
        or "Unknown CPU"
    )
    cpu_arch = (
        cpu_info.get("arch_string_raw")
        or cpu_info.get("arch")
        or platform.machine()
        or "x86_64"
    )
    raw_flags = [str(f).lower() for f in cpu_info.get("flags", [])]

    has_avx512 = any(
        f in raw_flags for f in ("avx512vnni", "avx512_vnni", "avx512f")
    )
    has_avx2 = "avx2" in raw_flags
    has_neon = "neon" in raw_flags or any(
        arm_token in cpu_arch.lower() for arm_token in ("arm", "aarch64")
    )

    # Detect GPUs
    gpus = detect_gpus()

    # Disk Space Check
    disk_target = target_disk_path if target_disk_path is not None else Path.home()
    check_path = disk_target
    while not check_path.exists() and check_path.parent != check_path:
        check_path = check_path.parent
    try:
        disk_usage = psutil.disk_usage(str(check_path))
        disk_free_gb = round(disk_usage.free / (1024**3), 2)
    except OSError:
        disk_free_gb = 0.0

    # Determine backend target based on selection hierarchy
    # 1. NVIDIA dGPU (>4000MB) -> "cuda"
    # 2. Apple Silicon -> "metal"
    # 3. AMD dGPU (>4000MB) -> "rocm"
    # 4. Fallback -> "cpu"
    backend_target = "cpu"
    discrete_nvidia = [
        g for g in gpus if g.vendor == "nvidia" and g.vram_total_mb >= 4000.0
    ]
    discrete_amd = [
        g for g in gpus if g.vendor == "amd" and g.vram_total_mb >= 4000.0
    ]
    apple_gpus = [g for g in gpus if g.vendor == "apple"]

    if discrete_nvidia:
        backend_target = "cuda"
    elif apple_gpus or (os_name == "Darwin" and has_neon):
        backend_target = "metal"
    elif discrete_amd:
        backend_target = "rocm"
    else:
        backend_target = "cpu"

    # Bandwidth heuristic
    bandwidth_gb_s = get_memory_bandwidth_gb_s(cpu_brand, cpu_arch, gpus)

    return HardwareProfile(
        os_name=os_name,
        cpu_brand=cpu_brand,
        cpu_arch=cpu_arch,
        physical_cores=physical_cores,
        logical_cores=logical_cores,
        has_avx512=has_avx512,
        has_avx2=has_avx2,
        has_neon=has_neon,
        ram_total_mb=ram_total_mb,
        ram_avail_mb=ram_avail_mb,
        disk_free_gb=disk_free_gb,
        gpus=gpus,
        bandwidth_gb_s=bandwidth_gb_s,
        backend_target=backend_target,
    )


def check_disk_space_preflight(
    target_path: Path, required_model_bytes: int, is_installer_build: bool = True
) -> Tuple[bool, float, float]:
    """Verify disk space sufficiency before downloading or packaging.

    Ratios:
    - Installer Build: 2.5x model weights (staging + bundle archive + extraction)
    - Download: 1.5x model weights (.part stream + destination)

    Returns:
    (has_space, free_gb, required_gb)
    """
    ratio = 2.5 if is_installer_build else 1.5
    required_bytes = int(required_model_bytes * ratio)

    check_path = target_path
    while not check_path.exists() and check_path.parent != check_path:
        check_path = check_path.parent

    try:
        disk_usage = psutil.disk_usage(str(check_path))
        free_bytes = disk_usage.free
    except OSError:
        free_bytes = 0

    free_gb = round(free_bytes / (1024**3), 2)
    required_gb = round(required_bytes / (1024**3), 2)
    has_space = free_bytes >= required_bytes

    return (has_space, free_gb, required_gb)
