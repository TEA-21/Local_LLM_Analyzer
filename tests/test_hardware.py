"""Unit tests for companion_packager.hardware subsystem.

Tests SIMD feature detection, zero-PyTorch GPU detection (nvidia-smi, Apple Silicon,
Windows CIM + Registry, Linux DRM), memory bandwidth heuristics, and disk space preflight.
"""

from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import pytest

from companion_packager.hardware.detector import (
    check_disk_space_preflight,
    detect_hardware,
    get_memory_bandwidth_gb_s,
)
from companion_packager.hardware.gpu_detector import (
    detect_apple_silicon_gpu,
    detect_gpus,
    detect_nvidia_gpus,
    detect_windows_gpus,
    parse_nvidia_smi,
    parse_registry_qwmemory,
)
from companion_packager.schemas import GPUInfo, HardwareProfile


def test_live_detect_hardware():
    """Verify live host probe returns valid HardwareProfile without crashing."""
    profile = detect_hardware()
    assert isinstance(profile, HardwareProfile)
    assert profile.os_name in ("Windows", "Darwin", "Linux")
    assert profile.ram_total_mb > 0.0
    assert profile.ram_avail_mb > 0.0
    assert profile.physical_cores >= 1
    assert profile.logical_cores >= profile.physical_cores
    assert profile.bandwidth_gb_s >= 25.0
    assert profile.backend_target in ("cpu", "cuda", "metal", "rocm")
    assert isinstance(profile.has_avx512, bool)
    assert isinstance(profile.has_avx2, bool)
    assert isinstance(profile.has_neon, bool)


def test_parse_nvidia_smi_single_gpu():
    """Test parsing standard nvidia-smi CSV single GPU line."""
    csv_data = "0, NVIDIA GeForce RTX 4090, 24564, 22000, 552.22\n"
    gpus = parse_nvidia_smi(csv_data)
    assert len(gpus) == 1
    gpu = gpus[0]
    assert gpu.index == 0
    assert gpu.name == "NVIDIA GeForce RTX 4090"
    assert gpu.vendor == "nvidia"
    assert gpu.vram_total_mb == 24564.0
    assert gpu.vram_free_mb == 22000.0
    assert gpu.driver_version == "552.22"


def test_parse_nvidia_smi_multi_gpu():
    """Test parsing multi-GPU output with different VRAM amounts."""
    csv_data = (
        "0, NVIDIA GeForce RTX 3060, 12288, 11000, 535.12\n"
        "1, NVIDIA GeForce RTX 4090, 24564, 23000, 552.22\n"
    )
    gpus = parse_nvidia_smi(csv_data)
    assert len(gpus) == 2
    assert gpus[0].name == "NVIDIA GeForce RTX 3060"
    assert gpus[1].name == "NVIDIA GeForce RTX 4090"
    # Max VRAM selection
    primary = max(gpus, key=lambda g: g.vram_total_mb)
    assert primary.vram_total_mb == 24564.0
    assert primary.index == 1


def test_parse_nvidia_smi_malformed_lines():
    """Test malformed and empty lines are skipped safely."""
    csv_data = (
        "\n"
        "garbage data\n"
        "0, NVIDIA RTX 3080, not_a_number, 8000, 530.0\n"
        "1, NVIDIA GeForce RTX 3080, 10240, 9000, 530.0\n"
    )
    gpus = parse_nvidia_smi(csv_data)
    assert len(gpus) == 1
    assert gpus[0].index == 1
    assert gpus[0].vram_total_mb == 10240.0


def test_detect_nvidia_gpus_subprocess_mock():
    """Test detect_nvidia_gpus calling subprocess."""
    mock_res = MagicMock()
    mock_res.returncode = 0
    mock_res.stdout = "0, NVIDIA RTX A6000, 49152, 45000, 535.98\n"

    with patch("subprocess.run", return_value=mock_res):
        gpus = detect_nvidia_gpus()
        assert len(gpus) == 1
        assert gpus[0].name == "NVIDIA RTX A6000"
        assert gpus[0].vram_total_mb == 49152.0

    # Test FileNotFoundError (nvidia-smi not present)
    with patch("subprocess.run", side_effect=FileNotFoundError):
        assert detect_nvidia_gpus() == []


def test_parse_registry_qwmemory_int_and_bytes():
    """Test registry qwMemorySize parser with int and 8-byte little-endian binary."""
    # 1. Standard integer REG_DWORD / REG_QWORD
    assert parse_registry_qwmemory(4294967296) == 4294967296
    assert parse_registry_qwmemory(8589934592) == 8589934592

    # 2. REG_BINARY 8-byte little-endian bytes for 8 GB (8589934592 bytes)
    # 8589934592 in hex is 0x02_0000_0000 -> little endian: 00 00 00 00 02 00 00 00
    val_bytes = (8589934592).to_bytes(8, byteorder="little")
    assert parse_registry_qwmemory(val_bytes) == 8589934592

    # 3. REG_BINARY for 16 GB (17179869184 bytes)
    val_16gb_bytes = (17179869184).to_bytes(8, byteorder="little")
    assert parse_registry_qwmemory(val_16gb_bytes) == 17179869184


def test_windows_gpus_cim_and_registry_mock():
    """Test Windows GPU detection with PowerShell CIM output and registry VRAM enhancement."""
    mock_cim_json = """
    [
        {
            "Name": "NVIDIA GeForce RTX 4070 Ti",
            "AdapterRAM": 4294967295,
            "DriverVersion": "31.0.15.5222"
        }
    ]
    """
    mock_res = MagicMock()
    mock_res.returncode = 0
    mock_res.stdout = mock_cim_json

    # 12 GB = 12 * 1024**3 = 12884901888 bytes
    mock_reg_map = {"NVIDIA GeForce RTX 4070 Ti": 12884901888}

    with patch("sys.platform", "win32"), \
         patch("subprocess.run", return_value=mock_res), \
         patch("companion_packager.hardware.gpu_detector.query_windows_registry_vram", return_value=mock_reg_map):
        gpus = detect_windows_gpus()
        assert len(gpus) == 1
        gpu = gpus[0]
        assert gpu.name == "NVIDIA GeForce RTX 4070 Ti"
        assert gpu.vendor == "nvidia"
        assert gpu.vram_total_mb == pytest.approx(12288.0, rel=1e-2)


def test_apple_silicon_detection_mock():
    """Test Apple Silicon detection mocking macOS sysctl output."""
    with patch("sys.platform", "darwin"), \
         patch("platform.machine", return_value="arm64"):

        def mock_sysctl(cmd, **kwargs):
            res = MagicMock()
            res.returncode = 0
            if "hw.memsize" in cmd:
                # 32 GB = 34359738368 bytes
                res.stdout = "34359738368\n"
            elif "machdep.cpu.brand_string" in cmd:
                res.stdout = "Apple M2 Max\n"
            else:
                res.stdout = ""
            return res

        with patch("subprocess.run", side_effect=mock_sysctl):
            gpus = detect_apple_silicon_gpu()
            assert len(gpus) == 1
            gpu = gpus[0]
            assert gpu.vendor == "apple"
            assert gpu.name == "Apple M2 Max"
            assert gpu.vram_total_mb == pytest.approx(32768.0, rel=1e-2)
            assert gpu.vram_free_mb == pytest.approx(32768.0, rel=1e-2)
            assert gpu.driver_version == "Metal"


def test_bandwidth_heuristics_gpu():
    """Test memory bandwidth heuristics for discrete GPUs."""
    # RTX 4090 / 3090 -> 1000.0 GB/s
    gpu_4090 = [GPUInfo(index=0, name="NVIDIA GeForce RTX 4090", vendor="nvidia", vram_total_mb=24564.0)]
    assert get_memory_bandwidth_gb_s("Intel Core i9", "x86_64", gpu_4090) == 1000.0

    # RTX 4080 / 3080 -> 750.0 GB/s
    gpu_3080 = [GPUInfo(index=0, name="NVIDIA GeForce RTX 3080", vendor="nvidia", vram_total_mb=10240.0)]
    assert get_memory_bandwidth_gb_s("Intel Core i7", "x86_64", gpu_3080) == 750.0

    # RTX 4070 / 3070 -> 500.0 GB/s
    gpu_4070 = [GPUInfo(index=0, name="NVIDIA GeForce RTX 4070", vendor="nvidia", vram_total_mb=12288.0)]
    assert get_memory_bandwidth_gb_s("AMD Ryzen 7", "x86_64", gpu_4070) == 500.0

    # RTX 4060 / 3060 -> 300.0 GB/s
    gpu_3060 = [GPUInfo(index=0, name="NVIDIA GeForce RTX 3060", vendor="nvidia", vram_total_mb=12288.0)]
    assert get_memory_bandwidth_gb_s("Intel Core i5", "x86_64", gpu_3060) == 300.0

    # Generic discrete GPU >= 4GB -> 250.0 GB/s
    gpu_amd = [GPUInfo(index=0, name="AMD Radeon RX 6700 XT", vendor="amd", vram_total_mb=12288.0)]
    assert get_memory_bandwidth_gb_s("AMD Ryzen 5", "x86_64", gpu_amd) == 250.0


def test_bandwidth_heuristics_apple_silicon():
    """Test memory bandwidth heuristics for Apple Silicon tiers."""
    # Ultra -> 800.0 GB/s
    gpu_ultra = [GPUInfo(index=0, name="Apple M2 Ultra", vendor="apple", vram_total_mb=65536.0)]
    assert get_memory_bandwidth_gb_s("Apple M2 Ultra", "arm64", gpu_ultra) == 800.0

    # Max -> 350.0 GB/s
    gpu_max = [GPUInfo(index=0, name="Apple M3 Max", vendor="apple", vram_total_mb=36864.0)]
    assert get_memory_bandwidth_gb_s("Apple M3 Max", "arm64", gpu_max) == 350.0

    # Pro -> 175.0 GB/s
    gpu_pro = [GPUInfo(index=0, name="Apple M1 Pro", vendor="apple", vram_total_mb=16384.0)]
    assert get_memory_bandwidth_gb_s("Apple M1 Pro", "arm64", gpu_pro) == 175.0

    # Base -> 100.0 GB/s
    gpu_base = [GPUInfo(index=0, name="Apple M2", vendor="apple", vram_total_mb=8192.0)]
    assert get_memory_bandwidth_gb_s("Apple M2", "arm64", gpu_base) == 100.0


def test_bandwidth_heuristics_cpu_tiers():
    """Test memory bandwidth heuristics for CPU RAM architecture tiers."""
    # DDR5 modern Intel 13th Gen -> 50.0 GB/s
    assert get_memory_bandwidth_gb_s("13th Gen Intel(R) Core(TM) i7-13700K", "x86_64", []) == 50.0
    # DDR5 modern AMD Zen 4 -> 50.0 GB/s
    assert get_memory_bandwidth_gb_s("AMD Ryzen 9 7950X 16-Core Processor", "x86_64", []) == 50.0
    # DDR5 explicit keyword -> 50.0 GB/s
    assert get_memory_bandwidth_gb_s("Intel Core i7 DDR5 System", "x86_64", []) == 50.0

    # DDR4 Intel 11th Gen (Reference Target) -> 30.0 GB/s
    assert get_memory_bandwidth_gb_s("11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz", "x86_64", []) == 30.0
    # DDR4 AMD Zen 3 -> 30.0 GB/s
    assert get_memory_bandwidth_gb_s("AMD Ryzen 7 5800X 8-Core Processor", "x86_64", []) == 30.0
    # DDR4 explicit keyword -> 30.0 GB/s
    assert get_memory_bandwidth_gb_s("AMD Ryzen Custom DDR4", "x86_64", []) == 30.0

    # Legacy CPU fallback -> 25.0 GB/s
    assert get_memory_bandwidth_gb_s("Intel(R) Core(TM) i7-4770K CPU @ 3.50GHz", "x86_64", []) == 25.0


def test_check_disk_space_preflight_build_and_download(tmp_path: Path):
    """Test disk space preflight enforcement with 2.5x build vs 1.5x download ratios."""
    # 1 GB model weights = 1024**3 bytes
    model_bytes = 1024**3

    # Mock psutil.disk_usage to return 2.0 GB free disk
    mock_usage = MagicMock()
    mock_usage.free = int(2.0 * 1024**3)

    with patch("psutil.disk_usage", return_value=mock_usage):
        # 1. Build mode (2.5x required = 2.5 GB): 2.0 GB is insufficient -> (False, 2.0, 2.5)
        has_space, free_gb, req_gb = check_disk_space_preflight(
            tmp_path, model_bytes, is_installer_build=True
        )
        assert not has_space
        assert free_gb == pytest.approx(2.0, rel=1e-2)
        assert req_gb == pytest.approx(2.5, rel=1e-2)

        # 2. Download mode (1.5x required = 1.5 GB): 2.0 GB is sufficient -> (True, 2.0, 1.5)
        has_space_dl, free_gb_dl, req_gb_dl = check_disk_space_preflight(
            tmp_path, model_bytes, is_installer_build=False
        )
        assert has_space_dl
        assert free_gb_dl == pytest.approx(2.0, rel=1e-2)
        assert req_gb_dl == pytest.approx(1.5, rel=1e-2)


def test_detect_hardware_backend_target_selection():
    """Test backend_target selection hierarchy across hardware profiles."""
    mock_vm = MagicMock()
    mock_vm.total = 16 * 1024**3
    mock_vm.available = 12 * 1024**3

    mock_disk = MagicMock()
    mock_disk.free = 50 * 1024**3

    # 1. CUDA when NVIDIA GPU has >= 4GB
    with patch("psutil.virtual_memory", return_value=mock_vm), \
         patch("psutil.disk_usage", return_value=mock_disk), \
         patch("companion_packager.hardware.detector.detect_gpus") as mock_gpus, \
         patch("cpuinfo.get_cpu_info", return_value={"flags": ["avx2"]}):
        mock_gpus.return_value = [
            GPUInfo(index=0, name="RTX 4060", vendor="nvidia", vram_total_mb=8192.0)
        ]
        profile = detect_hardware()
        assert profile.backend_target == "cuda"

    # 2. ROCm when AMD GPU has >= 4GB
    with patch("psutil.virtual_memory", return_value=mock_vm), \
         patch("psutil.disk_usage", return_value=mock_disk), \
         patch("companion_packager.hardware.detector.detect_gpus") as mock_gpus, \
         patch("cpuinfo.get_cpu_info", return_value={"flags": ["avx2"]}):
        mock_gpus.return_value = [
            GPUInfo(index=0, name="RX 7800 XT", vendor="amd", vram_total_mb=16384.0)
        ]
        profile = detect_hardware()
        assert profile.backend_target == "rocm"

    # 3. Metal on Apple Silicon
    with patch("platform.system", return_value="Darwin"), \
         patch("psutil.virtual_memory", return_value=mock_vm), \
         patch("psutil.disk_usage", return_value=mock_disk), \
         patch("companion_packager.hardware.detector.detect_gpus") as mock_gpus, \
         patch("cpuinfo.get_cpu_info", return_value={"flags": ["neon"], "arch_string_raw": "arm64"}):
        mock_gpus.return_value = [
            GPUInfo(index=0, name="Apple M2", vendor="apple", vram_total_mb=16384.0)
        ]
        profile = detect_hardware()
        assert profile.backend_target == "metal"

    # 4. CPU fallback when integrated GPU or < 4GB
    with patch("psutil.virtual_memory", return_value=mock_vm), \
         patch("psutil.disk_usage", return_value=mock_disk), \
         patch("companion_packager.hardware.detector.detect_gpus") as mock_gpus, \
         patch("cpuinfo.get_cpu_info", return_value={"flags": ["avx512vnni", "avx2"]}):
        mock_gpus.return_value = [
            GPUInfo(index=0, name="Intel Iris Xe", vendor="intel", vram_total_mb=2048.0)
        ]
        profile = detect_hardware()
        assert profile.backend_target == "cpu"
        assert profile.has_avx512 is True
