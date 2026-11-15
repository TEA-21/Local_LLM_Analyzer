"""Unit tests for Agent Companion configurator, bootstrap script generation, and path resolution."""

import os
import sys
from pathlib import Path

import pytest

from companion_packager.installer.configurator import (
    create_desktop_shortcut,
    inject_agent_companion_config,
    resolve_companion_path,
)
from companion_packager.schemas import CandidateModel, HardwareProfile


@pytest.fixture
def mock_hardware():
    return HardwareProfile(
        os_name="Windows",
        cpu_brand="AMD Ryzen 9 7950X",
        cpu_arch="x86_64",
        physical_cores=16,
        logical_cores=32,
        has_avx512=True,
        has_avx2=True,
        has_neon=False,
        ram_total_mb=65536.0,
        ram_avail_mb=48000.0,
        disk_free_gb=500.0,
        gpus=[],
        bandwidth_gb_s=50.0,
        backend_target="cuda",
    )


@pytest.fixture
def mock_model():
    return CandidateModel(
        repo_id="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        filename="qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        quant_method="Q4_K_M",
        param_count_billions=7.61,
        size_mb=4680.0,
        context_window=32768,
        arch_type="dense",
    )


def test_inject_agent_companion_config_creates_all_files(tmp_path, mock_hardware, mock_model):
    """Verify injection creates all 4 configuration and bootstrap files."""
    paths = inject_agent_companion_config(
        bundle_dir=tmp_path,
        selected_model=mock_model,
        hardware=mock_hardware,
        host="127.0.0.1",
        port=8080,
    )

    assert set(paths.keys()) == {"env", "py", "bat", "sh"}
    for key, file_path in paths.items():
        assert file_path.exists(), f"File {key} at {file_path} does not exist"


def test_agent_companion_env_keys_and_values(tmp_path, mock_hardware, mock_model):
    """Verify exact environment keys and quoted values in agent_companion.env."""
    paths = inject_agent_companion_config(
        bundle_dir=tmp_path,
        selected_model=mock_model,
        hardware=mock_hardware,
        host="0.0.0.0",
        port=9000,
    )

    env_text = paths["env"].read_text(encoding="utf-8")

    assert 'AGENT_COMPANION_MODEL_PATH="models/qwen2.5-coder-7b-instruct-q4_k_m.gguf"' in env_text
    assert 'AGENT_COMPANION_DEVICE="cuda"' in env_text
    assert 'AGENT_COMPANION_CONTEXT_WINDOW="32768"' in env_text
    assert 'AGENT_COMPANION_SSE_HOST="0.0.0.0"' in env_text
    assert 'AGENT_COMPANION_SSE_PORT="9000"' in env_text
    assert 'AGENT_COMPANION_THREADS="16"' in env_text
    assert 'AGENT_COMPANION_QUANT="Q4_K_M"' in env_text


def test_start_companion_bat_syntax(tmp_path, mock_hardware, mock_model):
    """Verify Windows batch script uses %%~j for quote stripping."""
    paths = inject_agent_companion_config(
        bundle_dir=tmp_path,
        selected_model=mock_model,
        hardware=mock_hardware,
    )

    bat_text = paths["bat"].read_text(encoding="utf-8")

    assert "@echo off" in bat_text
    assert 'for /F "usebackq eol=# tokens=1* delims==" %%i in ("%~dp0agent_companion.env") do (' in bat_text
    assert 'set "%%i=%%~j"' in bat_text
    assert '"%~dp0runtime\\python.exe" "%~dp0start_companion.py"' in bat_text


def test_start_companion_sh_syntax_and_mode(tmp_path, mock_hardware, mock_model):
    """Verify POSIX shell script contains correct export logic and executable bit."""
    paths = inject_agent_companion_config(
        bundle_dir=tmp_path,
        selected_model=mock_model,
        hardware=mock_hardware,
    )

    sh_text = paths["sh"].read_text(encoding="utf-8")

    assert "#!/usr/bin/env bash" in sh_text
    assert "set -e" in sh_text
    assert 'export $(grep -v \'^#\' "$SCRIPT_DIR/agent_companion.env" | xargs)' in sh_text
    assert 'exec "$SCRIPT_DIR/runtime/bin/python3" "$SCRIPT_DIR/start_companion.py"' in sh_text

    # Verify LF newlines
    raw_bytes = paths["sh"].read_bytes()
    assert b"\r\n" not in raw_bytes


def test_start_companion_py_compiles_and_parses_env(tmp_path, mock_hardware, mock_model, monkeypatch):
    """Verify generated Python launcher script is syntactically valid and strips quotes accurately."""
    paths = inject_agent_companion_config(
        bundle_dir=tmp_path,
        selected_model=mock_model,
        hardware=mock_hardware,
    )

    py_text = paths["py"].read_text(encoding="utf-8")

    # 1. Compilation check
    compiled = compile(py_text, str(paths["py"]), "exec")
    assert compiled is not None

    # 2. Namespace execution test for load_env
    namespace = {}
    exec(compiled, namespace)
    load_env = namespace["load_env"]

    # Clear target env vars
    for var in [
        "AGENT_COMPANION_MODEL_PATH",
        "AGENT_COMPANION_DEVICE",
        "AGENT_COMPANION_CONTEXT_WINDOW",
        "AGENT_COMPANION_SSE_HOST",
        "AGENT_COMPANION_SSE_PORT",
        "AGENT_COMPANION_THREADS",
        "AGENT_COMPANION_QUANT",
    ]:
        monkeypatch.delenv(var, raising=False)

    load_env(paths["env"])

    # Verify quote-stripped values loaded into os.environ
    assert os.environ["AGENT_COMPANION_MODEL_PATH"] == "models/qwen2.5-coder-7b-instruct-q4_k_m.gguf"
    assert os.environ["AGENT_COMPANION_DEVICE"] == "cuda"
    assert os.environ["AGENT_COMPANION_CONTEXT_WINDOW"] == "32768"
    assert os.environ["AGENT_COMPANION_SSE_HOST"] == "127.0.0.1"
    assert os.environ["AGENT_COMPANION_SSE_PORT"] == "8000"
    assert os.environ["AGENT_COMPANION_THREADS"] == "16"
    assert os.environ["AGENT_COMPANION_QUANT"] == "Q4_K_M"


def test_resolve_companion_path_explicit(tmp_path):
    """Test resolution with explicit valid and invalid paths."""
    resolved = resolve_companion_path(tmp_path)
    assert resolved == tmp_path.resolve()

    with pytest.raises(FileNotFoundError):
        resolve_companion_path(tmp_path / "nonexistent_dir")


def test_resolve_companion_path_env_var(tmp_path, monkeypatch):
    """Test resolution via AGENT_COMPANION_PATH environment variable."""
    fake_companion = tmp_path / "Agent Companion"
    fake_companion.mkdir()

    monkeypatch.setenv("AGENT_COMPANION_PATH", str(fake_companion))
    resolved = resolve_companion_path()
    assert resolved == fake_companion.resolve()


def test_resolve_companion_path_not_found(monkeypatch):
    """Test resolution raises FileNotFoundError when no candidates match."""
    monkeypatch.delenv("AGENT_COMPANION_PATH", raising=False)
    monkeypatch.setattr(Path, "exists", lambda self: False)

    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_companion_path()

    assert "Agent Companion directory not found" in str(exc_info.value)


def test_create_desktop_shortcut_fail_soft(tmp_path):
    """Test desktop shortcut creation returns None or path without raising unhandled exceptions."""
    result = create_desktop_shortcut(tmp_path)
    # Fail-soft check: must return None or a Path, never raise
    assert result is None or isinstance(result, Path)
