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
