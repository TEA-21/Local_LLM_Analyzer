"""Unit tests for POSIX self-extracting archive (.run) builder."""

import io
import stat
import tarfile
from pathlib import Path

from companion_packager.installer.sfx_builder import build_posix_sfx


def test_build_posix_sfx_structure(tmp_path: Path):
    """Test POSIX SFX generation: bash header followed by valid tar.gz payload."""
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    runtime_dir = bundle_dir / "runtime" / "bin"
    runtime_dir.mkdir(parents=True)
    (runtime_dir / "python3").write_text("#!/bin/sh\necho python", encoding="utf-8")

    wheels_dir = bundle_dir / "wheels"
    wheels_dir.mkdir()
    (wheels_dir / "mock_pkg-0.1.0-py3-none-any.whl").write_bytes(b"mock wheel bytes")

    output_run = tmp_path / "dist" / "AgentCompanion-Installer.run"

    res_path = build_posix_sfx(bundle_dir, output_run, app_name="Agent Companion")

    assert res_path == output_run
    assert output_run.exists()

    content = output_run.read_bytes()

    # Verify bash shebang
    assert content.startswith(b"#!/usr/bin/env bash\n")

    # Verify sentinel line
    sentinel = b"__ARCHIVE_BELOW__\n"
    assert sentinel in content

    header_bytes, tar_bytes = content.split(sentinel, 1)
    header_text = header_bytes.decode("utf-8")

    # Verify header logic
    assert 'APP_NAME="Agent Companion"' in header_text
    assert 'DEFAULT_INSTALL_DIR="$HOME/.local/share/agent-companion"' in header_text
    assert 'sed -e \'1,/^__ARCHIVE_BELOW__/d\' "$0" | tar -xz -C "$INSTALL_DIR"' in header_text
    assert '--find-links="$INSTALL_DIR/wheels"' in header_text
    assert "shared_contracts" in header_text
    assert "companion_packager" in header_text
    assert 'ln -sf "$INSTALL_DIR/start_companion.sh"' in header_text

    # Verify tar.gz payload validity
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        member_names = tar.getnames()
        assert "wheels/mock_pkg-0.1.0-py3-none-any.whl" in member_names
        assert "runtime/bin/python3" in member_names
        assert "decompress_and_run.sh" in member_names
        assert "start_companion.sh" in member_names


def test_build_posix_sfx_custom_app_name(tmp_path: Path):
    """Test custom app name in SFX archive header."""
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "file.txt").write_text("content", encoding="utf-8")

    output_run = tmp_path / "dist" / "CustomCompanion.run"
    build_posix_sfx(bundle_dir, output_run, app_name="Super Assistant")

    content = output_run.read_bytes()
    sentinel = b"__ARCHIVE_BELOW__\n"
    header_text = content.split(sentinel, 1)[0].decode("utf-8")

    assert 'APP_NAME="Super Assistant"' in header_text
    assert '[$APP_NAME] Installing' in header_text
