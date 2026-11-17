"""Unit tests for companion_packager CLI application."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from companion_packager.cli import app
from companion_packager.schemas import CandidateModel, HardwareProfile

runner = CliRunner()


@pytest.fixture
def mock_hardware():
    return HardwareProfile(
        os_name="Windows",
        cpu_brand="Intel(R) Core(TM) i7-1165G7",
        cpu_arch="AMD64",
        physical_cores=4,
        logical_cores=8,
        has_avx512=True,
        has_avx2=True,
        has_neon=False,
        ram_total_mb=16384.0,
        ram_avail_mb=8192.0,
        disk_free_gb=50.0,
        bandwidth_gb_s=30.0,
        backend_target="cpu",
    )


def test_cli_version():
    """Verify version command output."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "companion-packager v0.1.0" in result.output


def test_cli_version_flag():
    """Verify --version flag output."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "companion-packager v0.1.0" in result.output


def test_cli_doctor(mock_hardware):
    """Verify doctor command runs diagnostic checks."""
    with patch("companion_packager.cli.detect_hardware", return_value=mock_hardware):
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0
        assert "Python Version" in result.output
        assert "CPU SIMD Acceleration" in result.output
        assert "System RAM" in result.output
        assert "Disk Space" in result.output


def test_cli_profile_table(mock_hardware):
    """Verify profile command renders table."""
    with patch("companion_packager.cli.detect_hardware", return_value=mock_hardware):
        result = runner.invoke(app, ["profile"])
        assert result.exit_code == 0
        assert "Host Hardware Profile" in result.output
        assert "AVX-512" in result.output


def test_cli_profile_json(mock_hardware):
    """Verify profile --json renders valid JSON."""
    with patch("companion_packager.cli.detect_hardware", return_value=mock_hardware):
        result = runner.invoke(app, ["profile", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["cpu_brand"] == "Intel(R) Core(TM) i7-1165G7"
        assert data["ram_total_mb"] == 16384.0


def test_cli_analyze_headless(tmp_path, mock_hardware):
    """Verify analyze command in headless offline mode with markdown output."""
    out_md = tmp_path / "RECOMMENDATIONS.md"
    with patch("companion_packager.cli.detect_hardware", return_value=mock_hardware):
        result = runner.invoke(
            app,
            [
                "analyze",
                "--headless",
                "--offline",
                "--task",
                "coding",
                "--priority",
                "balanced",
                "--context-len",
                "16384",
                "--output-md",
                str(out_md),
            ],
        )
        assert result.exit_code == 0
        assert "Candidate Model Sizing" in result.output
        assert "Top Recommended Model" in result.output
        assert out_md.exists()
        md_text = out_md.read_text(encoding="utf-8")
        assert "Agent Companion Model Recommendations" in md_text


def test_cli_recommend_alias(tmp_path, mock_hardware):
    """Verify recommend command alias."""
    out_dir = tmp_path / "recs"
    with patch("companion_packager.cli.detect_hardware", return_value=mock_hardware):
        result = runner.invoke(
            app,
            [
                "recommend",
                "--non-interactive",
                "--offline",
                "--task",
                "chat",
                "--priority",
                "speed",
                "--output-dir",
                str(out_dir),
            ],
        )
        assert result.exit_code == 0
        assert (out_dir / "RECOMMENDATIONS.md").exists()


def test_cli_analyze_with_search_and_no_live(mock_hardware):
    """Verify analyze command with --search and --no-live options."""
    with patch("companion_packager.cli.detect_hardware", return_value=mock_hardware):
        result = runner.invoke(
            app,
            [
                "analyze",
                "--headless",
                "--no-live",
                "--search",
                "coder",
                "--limit",
                "3",
            ],
        )
        assert result.exit_code == 0
        assert "Candidate Model Sizing" in result.output


def test_cli_recommend_with_search(tmp_path, mock_hardware):
    """Verify recommend command with --search and --no-live flags."""
    out_dir = tmp_path / "recs_search"
    with patch("companion_packager.cli.detect_hardware", return_value=mock_hardware):
        result = runner.invoke(
            app,
            [
                "recommend",
                "--non-interactive",
                "--no-live",
                "--search",
                "deepseek",
                "--output-dir",
                str(out_dir),
            ],
        )
        assert result.exit_code == 0
        assert (out_dir / "RECOMMENDATIONS.md").exists()


def test_cli_download_offline_exists(tmp_path):
    """Verify download command in offline mode when model file already exists."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True)
    model_file = models_dir / "qwen2.5-coder-7b.gguf"
    model_file.write_bytes(b"dummy-gguf-weights")

    result = runner.invoke(
        app,
        [
            "download",
            "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
            "qwen2.5-coder-7b.gguf",
            "--dest",
            str(models_dir),
            "--offline",
        ],
    )
    assert result.exit_code == 0
    assert "Verified offline model file" in result.output


def test_cli_download_with_mock(tmp_path):
    """Verify download command calls download_file."""
    models_dir = tmp_path / "models"
    with patch("companion_packager.cli.download_file") as mock_dl:
        mock_dl.return_value = models_dir / "test.gguf"
        result = runner.invoke(
            app,
            [
                "download",
                "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
                "test.gguf",
                "--dest",
                str(models_dir),
            ],
        )
        assert result.exit_code == 0
        assert mock_dl.called
        assert "Download complete" in result.output


def test_cli_build_help():
    """Verify build command help output."""
    result = runner.invoke(app, ["build", "--help"])
    assert result.exit_code == 0
    assert "--mode" in result.output
    assert "--format" in result.output
    assert "--output-dir" in result.output
    assert "--companion-path" in result.output


def test_cli_install_help():
    """Verify install command help output."""
    result = runner.invoke(app, ["install", "--help"])
    assert result.exit_code == 0
    assert "--bundle-path" in result.output
    assert "--target-dir" in result.output


def test_cli_recommend_scenarios(tmp_path, mock_hardware):
    """Verify recommend command with --scenario dedicated and --scenario parallel."""
    out_dir_ded = tmp_path / "recs_dedicated"
    with patch("companion_packager.cli.detect_hardware", return_value=mock_hardware):
        result_ded = runner.invoke(
            app,
            [
                "recommend",
                "--non-interactive",
                "--offline",
                "--scenario",
                "dedicated",
                "--output-dir",
                str(out_dir_ded),
            ],
        )
        assert result_ded.exit_code == 0
        assert "Operational Concurrency Scenarios & Hardware Sizing" in result_ded.output
        assert "Candidate Model Sizing & Recommendation Matrix" in result_ded.output
        assert (out_dir_ded / "RECOMMENDATIONS.md").exists()

    out_dir_par = tmp_path / "recs_parallel"
    with patch("companion_packager.cli.detect_hardware", return_value=mock_hardware):
        result_par = runner.invoke(
            app,
            [
                "recommend",
                "--non-interactive",
                "--offline",
                "--scenario",
                "parallel",
                "--output-dir",
                str(out_dir_par),
            ],
        )
        assert result_par.exit_code == 0
        assert "Operational Concurrency Scenarios & Hardware Sizing" in result_par.output
        assert (out_dir_par / "RECOMMENDATIONS.md").exists()


def test_cli_build_manifest(tmp_path, mock_hardware):
    """Verify build command writes installer_manifest.json with scenario, task, and model path."""
    companion_root = tmp_path / "fake_companion"
    companion_root.mkdir(parents=True)
    pkgs_dir = companion_root / "packages"
    pkgs_dir.mkdir(parents=True)
    pkg_a = pkgs_dir / "shared_contracts"
    pkg_a.mkdir(parents=True)
    (pkg_a / "pyproject.toml").write_text(
        '[project]\nname = "shared-contracts"\nversion = "0.1.0"\ndependencies = []\n',
        encoding="utf-8",
    )

    out_dist = tmp_path / "dist"

    with patch("companion_packager.cli.detect_hardware", return_value=mock_hardware), \
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
                "--scenario",
                "dedicated",
                "--output-dir",
                str(out_dist),
                "--companion-path",
                str(companion_root),
                "--offline",
            ],
        )

        assert result.exit_code == 0
        assert "Build Succeeded" in result.output
        manifest_file = out_dist / "bundle" / "installer_manifest.json"
        assert manifest_file.exists()
        manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
        assert manifest_data["scenario"] == "dedicated"
        assert manifest_data["task"] == "coding"
        assert manifest_data["priority"] == "balanced"
        assert "models/" in manifest_data["model_path"]


def test_cli_install_with_manifest(tmp_path):
    """Verify install command parses and applies installer_manifest.json."""
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(parents=True)
    wheels_dir = bundle_dir / "wheels"
    wheels_dir.mkdir(parents=True)
    runtime_dir = bundle_dir / "runtime"
    runtime_dir.mkdir(parents=True)
    models_dir = bundle_dir / "models"
    models_dir.mkdir(parents=True)

    dummy_model = models_dir / "dummy-model.gguf"
    dummy_model.write_bytes(b"dummy-gguf-weights")

    manifest_file = bundle_dir / "installer_manifest.json"
    manifest_file.write_text(
        json.dumps({
            "scenario": "minimal_interference",
            "task": "coding",
            "priority": "speed",
            "model_path": "models/dummy-model.gguf",
        }),
        encoding="utf-8",
    )

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
                "--manifest",
                str(manifest_file),
                "--non-interactive",
            ],
        )
        assert result.exit_code == 0
        assert "installed successfully" in result.output
        assert "Loaded installer manifest" in result.output
