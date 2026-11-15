"""Lightweight zero-PyTorch GPU detection subsystem.

Probes NVIDIA (nvidia-smi), Apple Silicon (sysctl/Metal), Windows CIM/registry,
and Linux sysfs without introducing heavy PyTorch or CUDA runtime dependencies.
"""

import json
import os
import platform
import subprocess
import sys
from typing import Any, List, Optional

from companion_packager.schemas import GPUInfo

REGISTRY_DISPLAY_CLASS = (
    r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
)


def parse_nvidia_smi(output: str) -> List[GPUInfo]:
    """Parse csv output from nvidia-smi query command.

    Expected columns: index, name, memory.total, memory.free, driver_version
    """
    gpus: List[GPUInfo] = []
    lines = [line.strip() for line in output.strip().splitlines() if line.strip()]
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            try:
                idx = int(parts[0])
                name = parts[1]
                vram_total = float(parts[2])
                vram_free = float(parts[3])
                driver_ver = parts[4]
                gpus.append(
                    GPUInfo(
                        index=idx,
                        name=name,
                        vendor="nvidia",
                        vram_total_mb=vram_total,
                        vram_free_mb=vram_free,
                        driver_version=driver_ver,
                    )
                )
            except (ValueError, IndexError):
                continue
    return gpus


def detect_nvidia_gpus() -> List[GPUInfo]:
    """Detect NVIDIA GPUs via nvidia-smi CLI."""
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if res.returncode == 0 and res.stdout:
            return parse_nvidia_smi(res.stdout)
    except (FileNotFoundError, subprocess.SubprocessError, OSError, ValueError):
        pass
    return []


def detect_apple_silicon_gpu() -> List[GPUInfo]:
    """Detect Apple Silicon unified memory GPU on macOS."""
    if sys.platform != "darwin":
        return []

    machine = platform.machine().lower()
    if machine not in ("arm64", "aarch64"):
        return []

    # Get total memory via sysctl -n hw.memsize
    total_mb = 0.0
    try:
        res = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            total_bytes = int(res.stdout.strip())
            total_mb = round(total_bytes / (1024 * 1024), 2)
    except (FileNotFoundError, subprocess.SubprocessError, OSError, ValueError):
        pass

    # Get brand name
    chip_name = "Apple Silicon"
    try:
        res = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            chip_name = res.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        pass

    return [
        GPUInfo(
            index=0,
            name=chip_name,
            vendor="apple",
            vram_total_mb=total_mb,
            vram_free_mb=total_mb,
            driver_version="Metal",
        )
    ]


def parse_registry_qwmemory(val: Any) -> int:
    """Parse HardwareInformation.qwMemorySize safely from registry.

    Handles REG_DWORD/REG_QWORD (int) and REG_BINARY (8-byte little-endian bytes).
    """
    if isinstance(val, (bytes, bytearray)):
        return int.from_bytes(val, "little")
    return int(val)


def query_windows_registry_vram() -> dict[str, int]:
    """Query Windows registry for discrete GPU 64-bit VRAM amounts.

    Returns mapping of driver description or subkey to VRAM bytes.
    """
    if sys.platform != "win32":
        return {}

    results: dict[str, int] = {}
    try:
        import winreg  # type: ignore

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, REGISTRY_DISPLAY_CLASS) as class_key:
            sub_idx = 0
            while True:
                try:
                    sub_name = winreg.EnumKey(class_key, sub_idx)
                    sub_idx += 1
                except OSError:
                    break

                try:
                    with winreg.OpenKey(class_key, sub_name) as sub_key:
                        qw_bytes = 0
                        try:
                            val, _ = winreg.QueryValueEx(
                                sub_key, "HardwareInformation.qwMemorySize"
                            )
                            qw_bytes = parse_registry_qwmemory(val)
                        except (FileNotFoundError, OSError, ValueError):
                            pass

                        desc = ""
                        try:
                            desc_val, _ = winreg.QueryValueEx(sub_key, "DriverDesc")
                            desc = str(desc_val)
                        except (FileNotFoundError, OSError):
                            desc = sub_name

                        if qw_bytes > 0:
                            results[desc] = qw_bytes
                            results[sub_name] = qw_bytes
                except (PermissionError, FileNotFoundError, OSError):
                    continue
    except (ImportError, PermissionError, FileNotFoundError, OSError):
        pass

    return results


def detect_windows_gpus() -> List[GPUInfo]:
    """Detect GPUs on Windows via PowerShell CIM and safe Registry query."""
    if sys.platform != "win32":
        return []

    gpus: List[GPUInfo] = []
    # 1. Query Win32_VideoController via PowerShell CIM
    cmd = (
        'Get-CimInstance Win32_VideoController | '
        'Select-Object Name, AdapterRAM, DriverVersion | '
        'ConvertTo-Json -Compress'
    )
    raw_controllers: list[dict[str, Any]] = []
    try:
        res = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            parsed = json.loads(res.stdout.strip())
            if isinstance(parsed, dict):
                raw_controllers = [parsed]
            elif isinstance(parsed, list):
                raw_controllers = parsed
    except (FileNotFoundError, subprocess.SubprocessError, OSError, json.JSONDecodeError):
        pass

    # 2. Query safe registry for 64-bit VRAM exceeding 4GB
    reg_vram_map = query_windows_registry_vram()

    for idx, ctrl in enumerate(raw_controllers):
        name = str(ctrl.get("Name") or f"Display Adapter {idx}")
        adapter_ram = ctrl.get("AdapterRAM") or 0
        driver_ver = str(ctrl.get("DriverVersion") or "")

        try:
            adapter_ram_int = int(adapter_ram)
        except (ValueError, TypeError):
            adapter_ram_int = 0

        # Check if registry has larger 64-bit VRAM for this GPU
        vram_bytes = adapter_ram_int
        for desc_key, reg_bytes in reg_vram_map.items():
            if desc_key.lower() in name.lower() or name.lower() in desc_key.lower():
                if reg_bytes > vram_bytes:
                    vram_bytes = reg_bytes
                    break

        vram_mb = round(max(vram_bytes, 0) / (1024 * 1024), 2)

        # Classify vendor
        lower_name = name.lower()
        if "nvidia" in lower_name or "geforce" in lower_name or "quadro" in lower_name:
            vendor = "nvidia"
        elif "amd" in lower_name or "radeon" in lower_name:
            vendor = "amd"
        elif "intel" in lower_name or "iris" in lower_name or "arc" in lower_name:
            vendor = "intel"
        else:
            vendor = "unknown"

        gpus.append(
            GPUInfo(
                index=idx,
                name=name,
                vendor=vendor,
                vram_total_mb=vram_mb,
                vram_free_mb=vram_mb,
                driver_version=driver_ver,
            )
        )

    return gpus


def detect_linux_gpus() -> List[GPUInfo]:
    """Detect GPUs on Linux via sysfs and DRM interfaces."""
    if not sys.platform.startswith("linux"):
        return []

    gpus: List[GPUInfo] = []
    import glob

    vram_files = glob.glob("/sys/class/drm/card*/device/mem_info_vram_total")
    for idx, vram_path in enumerate(vram_files):
        try:
            with open(vram_path, "r", encoding="utf-8") as f:
                vram_bytes = int(f.read().strip())
                vram_mb = round(vram_bytes / (1024 * 1024), 2)

            name = f"DRM Card {idx}"
            vendor = "unknown"
            # Try to identify vendor from device/vendor file
            dev_dir = os.path.dirname(vram_path)
            vendor_path = os.path.join(dev_dir, "vendor")
            if os.path.exists(vendor_path):
                with open(vendor_path, "r", encoding="utf-8") as vf:
                    vid = vf.read().strip().lower()
                    if "0x10de" in vid:
                        vendor = "nvidia"
                    elif "0x1002" in vid:
                        vendor = "amd"
                    elif "0x8086" in vid:
                        vendor = "intel"

            gpus.append(
                GPUInfo(
                    index=idx,
                    name=name,
                    vendor=vendor,
                    vram_total_mb=vram_mb,
                    vram_free_mb=vram_mb,
                )
            )
        except (OSError, ValueError):
            continue

    return gpus


def detect_gpus() -> List[GPUInfo]:
    """Zero-PyTorch multi-platform GPU detection.

    Probes NVIDIA CLI first across all platforms. If no NVIDIA GPUs found,
    delegates to platform-specific detection (macOS Apple Silicon, Windows CIM/Registry,
    Linux DRM).
    """
    # 1. Try nvidia-smi first (works cross-platform if CUDA/NVIDIA installed)
    nvidia_gpus = detect_nvidia_gpus()
    if nvidia_gpus:
        return nvidia_gpus

    # 2. macOS Apple Silicon
    if sys.platform == "darwin":
        apple_gpus = detect_apple_silicon_gpu()
        if apple_gpus:
            return apple_gpus

    # 3. Windows PowerShell CIM & Registry
    if sys.platform == "win32":
        win_gpus = detect_windows_gpus()
        if win_gpus:
            return win_gpus

    # 4. Linux DRM sysfs
    if sys.platform.startswith("linux"):
        linux_gpus = detect_linux_gpus()
        if linux_gpus:
            return linux_gpus

    return []
