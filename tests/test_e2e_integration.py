"""Comprehensive End-to-End Integration Test Harness for companion-packager.

Simulates complete user lifecycle:
1. Probe host hardware.
2. Interview / specify user needs.
3. Recommend top models from airgapped benchmarks cache.
4. Create temporary bundle directory.
5. Inject Agent Companion configuration (agent_companion.env, start_companion.bat,
   start_companion.sh, start_companion.py).
6. Stage monorepo package wheel builds and portable installer zip generation.
7. Verify generated .env contains valid device target and model path.
8. Verify start_companion.bat contains 'set "%%i=%%~j"' quote stripping.
9. Verify portable bundle zip contains runtime, wheels, config files, and launchers.
10. Test offline installation to target directory.
"""

import os
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from companion_packager.cli import app
from companion_packager.hardware.detector import detect_hardware
from companion_packager.installer.configurator import (
    inject_agent_companion_config,
    resolve_companion_path,
)
from companion_packager.installer.inno_builder import build_windows_installer
from companion_packager.interview.questionnaire import ask_user_needs
from companion_packager.research.hf_client import get_model_candidates
from companion_packager.schemas import GPUInfo, HardwareProfile, UserNeeds
from companion_packager.sizing.recommender import rank_candidates

runner = CliRunner()


@pytest.fixture
def mock_cpu_hardware():
    return HardwareProfile(
        os_name="Windows",
        cpu_brand="Intel(R) Core(TM) i7-1165G7",
        cpu_arch="AMD64",
        physical_cores=4,
        logical_cores=8,
        has_avx512=True,
        has_avx2=True,
        has_neon=False,
        ram_total_mb=8192.0,
        ram_avail_mb=6000.0,
        disk_free_gb=40.0,
        bandwidth_gb_s=30.0,
        backend_target="cpu",
    )


@pytest.fixture
def mock_gpu_hardware():
    return HardwareProfile(
        os_name="Windows",
        cpu_brand="AMD Ryzen 9 7950X",
        cpu_arch="AMD64",
        physical_cores=16,
        logical_cores=32,
        has_avx512=True,
        has_avx2=True,
        has_neon=False,
        ram_total_mb=65536.0,
        ram_avail_mb=48000.0,
        disk_free_gb=200.0,
        bandwidth_gb_s=60.0,
        backend_target="cuda",
        gpus=[
            GPUInfo(
                index=0,
                name="NVIDIA GeForce RTX 4090",
                vendor="nvidia",
                vram_total_mb=24576.0,
                vram_free_mb=22000.0,
            )
        ],
    )


def test_e2e_user_workflow_cpu_pipeline(tmp_path, mock_cpu_hardware):
    """E2E Test 1: Full pipeline on simulated sub-6GB AVX-512 CPU laptop."""
    # Step 1: Probe host hardware
    hardware = mock_cpu_hardware
    assert hardware.backend_target == "cpu"
    assert hardware.has_avx512 is True

    # Step 2: User needs discovery (headless)
    needs = ask_user_needs(
        interactive=False,
        task="coding",
        priority="balanced",
        min_context_tokens=16384,
        require_permissive_license=False,
    )
    assert needs.task == "coding"
    assert needs.min_context_tokens == 16384

    # Step 3: Recommend top models from offline benchmarks cache
    candidates = get_model_candidates(offline=True)
    assert len(candidates) >= 5

    scorecards = rank_candidates(candidates, hardware, needs)
    assert len(scorecards) > 0
    top_scorecard = scorecards[0]
    top_model = top_scorecard.candidate
    assert top_model is not None

    # Step 4: Create bundle staging directory
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(parents=True)
    wheels_dir = bundle_dir / "wheels"
    wheels_dir.mkdir(parents=True)
    runtime_dir = bundle_dir / "runtime"
    runtime_dir.mkdir(parents=True)

    # Step 5: Inject Agent Companion configuration
    config_paths = inject_agent_companion_config(
        bundle_dir=bundle_dir,
        selected_model=top_model,
        hardware=hardware,
        host="127.0.0.1",
        port=8000,
    )

    env_path = config_paths["env"]
    bat_path = config_paths["bat"]
    sh_path = config_paths["sh"]
    py_path = config_paths["py"]

    assert env_path.exists()
    assert bat_path.exists()
    assert sh_path.exists()
    assert py_path.exists()

    # Step 6: Verify generated .env contains device target and model path
    env_content = env_path.read_text(encoding="utf-8")
    assert 'AGENT_COMPANION_DEVICE="cpu"' in env_content
    assert f'AGENT_COMPANION_MODEL_PATH="models/{top_model.filename}"' in env_content
    assert 'AGENT_COMPANION_SSE_HOST="127.0.0.1"' in env_content
    assert 'AGENT_COMPANION_SSE_PORT="8000"' in env_content

    # Step 7: Verify start_companion.bat has 'set "%%i=%%~j"' quote stripping
    bat_content = bat_path.read_text(encoding="utf-8")
    assert 'for /F "usebackq eol=# tokens=1* delims==" %%i in' in bat_content
    assert 'set "%%i=%%~j"' in bat_content

    # Step 8: Stage mock wheels and runtime
    (wheels_dir / "shared_contracts-0.1.0-py3-none-any.whl").write_bytes(b"dummy-whl")
    (wheels_dir / "model_runtime-0.1.0-py3-none-any.whl").write_bytes(b"dummy-whl")
    (wheels_dir / "security_server-0.1.0-py3-none-any.whl").write_bytes(b"dummy-whl")
    (wheels_dir / "document_pipeline-0.1.0-py3-none-any.whl").write_bytes(b"dummy-whl")
    (wheels_dir / "agent_tooling-0.1.0-py3-none-any.whl").write_bytes(b"dummy-whl")
    (wheels_dir / "companion_packager-0.1.0-py3-none-any.whl").write_bytes(b"dummy-whl")

    (runtime_dir / "python.exe").write_bytes(b"dummy-python-exe")

    # Step 9: Build portable installer zip
    dist_dir = tmp_path / "dist"
    with patch("companion_packager.installer.inno_builder.find_iscc", return_value=None):
        artifact = build_windows_installer(
            bundle_dir=bundle_dir,
            output_dir=dist_dir,
            app_name="Agent Companion",
            app_version="0.1.0",
        )

    assert artifact.exists()
    assert artifact.suffix == ".zip"
    assert "Portable" in artifact.name

    # Step 10: Verify portable bundle zip contents
    with zipfile.ZipFile(artifact, "r") as z:
        namelist = z.namelist()
        assert "agent_companion.env" in namelist
        assert "start_companion.bat" in namelist
        assert "start_companion.sh" in namelist
        assert "start_companion.py" in namelist
        assert "run_setup.bat" in namelist
        assert "runtime/python.exe" in namelist
        assert any("shared_contracts" in n for n in namelist)
        assert any("model_runtime" in n for n in namelist)


def test_e2e_user_workflow_gpu_pipeline(tmp_path, mock_gpu_hardware):
    """E2E Test 2: Full pipeline on high-tier RTX 4090 GPU workstation."""
    hardware = mock_gpu_hardware
    assert hardware.backend_target == "cuda"
    assert len(hardware.gpus) == 1

    needs = ask_user_needs(
        interactive=False,
        task="coding",
        priority="quality",
        min_context_tokens=32768,
        require_permissive_license=False,
    )

    candidates = get_model_candidates(offline=True)
    scorecards = rank_candidates(candidates, hardware, needs)
    assert len(scorecards) > 0

    top_model = scorecards[0].candidate
    bundle_dir = tmp_path / "gpu_bundle"
    bundle_dir.mkdir(parents=True)

    config_paths = inject_agent_companion_config(
        bundle_dir=bundle_dir,
        selected_model=top_model,
        hardware=hardware,
    )

    env_content = config_paths["env"].read_text(encoding="utf-8")
    assert 'AGENT_COMPANION_DEVICE="cuda"' in env_content
    assert top_model.filename in env_content


def test_e2e_cli_install_flow(tmp_path):
    """E2E Test 3: Test companion-packager install command from portable bundle."""
    bundle_dir = tmp_path / "test_bundle"
    bundle_dir.mkdir(parents=True)
    wheels_dir = bundle_dir / "wheels"
    wheels_dir.mkdir(parents=True)
    runtime_dir = bundle_dir / "runtime"
    runtime_dir.mkdir(parents=True)

    (bundle_dir / "agent_companion.env").write_text('AGENT_COMPANION_DEVICE="cpu"\n')
    (bundle_dir / "start_companion.bat").write_text('@echo off\necho starting\n')
    (bundle_dir / "start_companion.sh").write_text('#!/bin/bash\necho starting\n')
    (wheels_dir / "dummy.whl").write_bytes(b"dummy")
    (runtime_dir / "python.exe").write_bytes(b"dummy")

    target_install_dir = tmp_path / "installed_app"

    with patch("subprocess.run") as mock_subproc:
        mock_subproc.return_value.returncode = 0
        result = runner.invoke(
            app,
            [
                "install",
                "--bundle-path",
                str(bundle_dir),
                "--target-dir",
                str(target_install_dir),
                "--non-interactive",
            ],
        )
        assert result.exit_code == 0
        assert "installed successfully" in result.output
        assert (target_install_dir / "agent_companion.env").exists()
        assert (target_install_dir / "start_companion.bat").exists()
        assert (target_install_dir / "wheels" / "dummy.whl").exists()
        assert (target_install_dir / "runtime" / "python.exe").exists()


def test_e2e_cli_build_thin_mode(tmp_path, mock_cpu_hardware):
    """E2E Test 4: CLI build command in thin mode with mocked monorepo."""
    companion_root = tmp_path / "fake_companion"
    companion_root.mkdir(parents=True)
    pkgs_dir = companion_root / "packages"
    pkgs_dir.mkdir(parents=True)

    # Setup dummy package in packages
    pkg_a = pkgs_dir / "shared_contracts"
    pkg_a.mkdir(parents=True)
    (pkg_a / "pyproject.toml").write_text(
        '[project]\nname = "shared-contracts"\nversion = "0.1.0"\ndependencies = []\n',
        encoding="utf-8",
    )

    out_dist = tmp_path / "dist"

    with patch("companion_packager.cli.detect_hardware", return_value=mock_cpu_hardware), \
         patch("companion_packager.cli.stage_and_build_monorepo_wheels") as mock_monorepo, \
         patch("companion_packager.cli.build_packager_wheel") as mock_pkg_whl, \
         patch("companion_packager.cli.vendor_third_party_wheels") as mock_vendor, \
         patch("companion_packager.cli.acquire_python_standalone") as mock_python:

        mock_monorepo.return_value = [out_dist / "bundle" / "wheels" / "dummy.whl"]
        mock_pkg_whl.return_value = out_dist / "bundle" / "wheels" / "companion_packager-0.1.0-py3-none-any.whl"
        mock_vendor.return_value = []
        mock_python.return_value = out_dist / "bundle" / "runtime" / "python.exe"

        result = runner.invoke(
            app,
            [
                "build",
                "--mode",
                "thin",
                "--format",
                "portable",
                "--output-dir",
                str(out_dist),
                "--companion-path",
                str(companion_root),
                "--offline",
            ],
        )

        assert result.exit_code == 0
        assert "Build Succeeded" in result.output
        assert "THIN" in result.output
        assert (out_dist / "AgentCompanion-Windows-Portable.zip").exists()
