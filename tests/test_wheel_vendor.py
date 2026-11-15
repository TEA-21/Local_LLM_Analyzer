"""Unit tests for wheel vendorization and standalone Python acquisition."""

import io
import subprocess
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from companion_packager.installer.wheel_vendor import (
    acquire_python_standalone,
    extract_and_filter_requirements,
    stage_and_build_monorepo_wheels,
    vendor_third_party_wheels,
)


def test_extract_and_filter_requirements_mock(tmp_path: Path):
    """Test extracting dependencies from pyproject.toml and stripping blacklisted packages."""
    companion_path = tmp_path / "Agent Companion"
    packages_dir = companion_path / "packages"
    packages_dir.mkdir(parents=True)

    # Package A
    pkg_a = packages_dir / "pkg_a"
    pkg_a.mkdir()
    (pkg_a / "pyproject.toml").write_text(
        """[project]
name = "pkg-a"
version = "0.1.0"
dependencies = [
    "shared-contracts",
    "model-runtime",
    "pydantic>=2.0.0",
    "requests>=2.31.0",
]
""",
        encoding="utf-8",
    )

    # Package B
    pkg_b = packages_dir / "pkg_b"
    pkg_b.mkdir()
    (pkg_b / "pyproject.toml").write_text(
        """[project]
name = "pkg-b"
version = "0.1.0"
dependencies = [
    "agent_tooling",
    "security_server",
    "document-pipeline",
    "companion-packager",
    "httpx>=0.25.0",
]
""",
        encoding="utf-8",
    )

    reqs = extract_and_filter_requirements(companion_path)

    # Verify blacklisted workspace packages were removed
    assert not any("shared-contracts" in r for r in reqs)
    assert not any("model-runtime" in r for r in reqs)
    assert not any("agent_tooling" in r for r in reqs)
    assert not any("security_server" in r for r in reqs)
    assert not any("document-pipeline" in r for r in reqs)
    assert not any("companion-packager" in r for r in reqs)

    # Verify external third-party dependencies are kept
    assert any("requests>=2.31.0" in r for r in reqs)
    assert any("httpx" in r for r in reqs)

    # Verify companion-packager core dependencies were appended
    assert any("typer>=0.12.0" in r for r in reqs)
    assert any("rich>=13.7.0" in r for r in reqs)
    assert any("build>=1.2.0" in r for r in reqs)
    assert any("hatchling>=1.21.0" in r for r in reqs)

    # Verify sorted
    assert reqs == sorted(reqs)


def test_extract_and_filter_requirements_real_repo():
    """Verify extraction against real Agent Companion monorepo if present on disk."""
    real_path = Path(r"D:\Projects\Agent Companion")
    if not (real_path / "packages").exists():
        pytest.skip("Real Agent Companion repo not present at expected path")

    reqs = extract_and_filter_requirements(real_path)
    assert not any(r.startswith("shared-contracts") for r in reqs)
    assert not any(r.startswith("model-runtime") for r in reqs)
    assert any("fastapi>=0.110.0" in r for r in reqs)
    assert any("pydantic" in r for r in reqs)
    assert any("typer>=0.12.0" in r for r in reqs)


def test_stage_and_build_monorepo_wheels_isolated_staging(tmp_path: Path):
    """Test staging isolation: stub README.md is created only in staging, leaving source repo clean."""
    mock_companion = tmp_path / "Agent Companion"
    pkg_dir = mock_companion / "packages" / "shared_contracts"
    src_dir = pkg_dir / "src" / "shared_contracts"
    src_dir.mkdir(parents=True)

    (src_dir / "__init__.py").write_text('__version__ = "0.1.0"\n', encoding="utf-8")
    (pkg_dir / "pyproject.toml").write_text(
        """[project]
name = "shared-contracts"
version = "0.1.0"
description = "Mock shared contracts"
readme = "README.md"
requires-python = ">=3.11"
dependencies = []

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/shared_contracts"]
""",
        encoding="utf-8",
    )

    # Confirm source README.md does NOT exist
    assert not (pkg_dir / "README.md").exists()

    out_wheels = tmp_path / "wheels"
    built = stage_and_build_monorepo_wheels(mock_companion, out_wheels)

    # Source repository MUST remain untouched
    assert not (pkg_dir / "README.md").exists()

    # Output wheel must be produced
    assert len(built) == 1
    assert built[0].name.startswith("shared_contracts-0.1.0")
    assert built[0].exists()


def test_acquire_python_standalone_root_stripping(tmp_path: Path):
    """Test python-build-standalone archive extraction with top-level root directory stripping."""
    mock_archive = tmp_path / "python_standalone.tar.gz"

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        # Add python/python.exe
        data_exe = b"MZ mock windows python binary"
        ti_exe = tarfile.TarInfo(name="python/python.exe")
        ti_exe.size = len(data_exe)
        ti_exe.mode = 0o755
        tar.addfile(ti_exe, io.BytesIO(data_exe))

        # Add python/bin/python3
        data_sh = b"#!/bin/sh\necho mock python3"
        ti_sh = tarfile.TarInfo(name="python/bin/python3")
        ti_sh.size = len(data_sh)
        ti_sh.mode = 0o755
        tar.addfile(ti_sh, io.BytesIO(data_sh))

        # Add python/lib/test.py
        data_lib = b"print('lib')"
        ti_lib = tarfile.TarInfo(name="python/lib/test.py")
        ti_lib.size = len(data_lib)
        tar.addfile(ti_lib, io.BytesIO(data_lib))

    mock_archive.write_bytes(buf.getvalue())

    # Windows target extraction
    target_win = tmp_path / "runtime_win"
    res_win = acquire_python_standalone("windows", target_win, python_dist_path=mock_archive)
    assert res_win == target_win / "python.exe"
    assert (target_win / "python.exe").exists()
    assert (target_win / "lib" / "test.py").exists()
    # Confirm root prefix was stripped
    assert not (target_win / "python" / "python.exe").exists()

    # Linux target extraction
    target_linux = tmp_path / "runtime_linux"
    res_linux = acquire_python_standalone("linux", target_linux, python_dist_path=mock_archive)
    assert res_linux == target_linux / "bin" / "python3"
    assert (target_linux / "bin" / "python3").exists()


def test_vendor_third_party_wheels_cache_and_offline(tmp_path: Path):
    """Test offline wheel vendoring from cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "pydantic-2.7.0-py3-none-any.whl").write_bytes(b"mock whl")

    out_wheels = tmp_path / "out_wheels"

    # Offline mode with cache
    wheels = vendor_third_party_wheels(
        requirements=["pydantic>=2.7.0"],
        output_wheels_dir=out_wheels,
        wheels_cache_dir=cache_dir,
        offline=True,
    )
    assert len(wheels) == 1
    assert wheels[0].name == "pydantic-2.7.0-py3-none-any.whl"
    assert (out_wheels / "pydantic-2.7.0-py3-none-any.whl").exists()

    # Offline mode without cache raises ValueError
    empty_cache = tmp_path / "empty_cache"
    empty_cache.mkdir()
    empty_out = tmp_path / "empty_out"
    with patch("pathlib.Path.home", return_value=tmp_path / "fake_home"):
        with pytest.raises(ValueError, match="Offline mode requested but no wheels cache found"):
            vendor_third_party_wheels(
                requirements=["pydantic>=2.7.0"],
                output_wheels_dir=empty_out,
                wheels_cache_dir=empty_cache,
                offline=True,
            )


def test_vendor_third_party_wheels_online_invocation(tmp_path: Path):
    """Test online pip download command invocation with correct ABI and platform flags."""
    out_wheels = tmp_path / "wheels"

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="Successfully downloaded", stderr="")
        with patch("pathlib.Path.home", return_value=tmp_path / "fake_home"):
            vendor_third_party_wheels(
                requirements=["httpx>=0.27.0"],
                output_wheels_dir=out_wheels,
                target_os="linux",
                offline=False,
            )

        assert mock_run.called
        cmd = mock_run.call_args[0][0]
        assert "pip" in cmd
        assert "download" in cmd
        assert "--platform" in cmd
        assert "manylinux2014_x86_64" in cmd
        assert "--abi" in cmd
        assert "cp311" in cmd
        assert "--only-binary=:all:" in cmd
