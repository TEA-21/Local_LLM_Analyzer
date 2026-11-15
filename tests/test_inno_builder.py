"""Unit tests for Inno Setup compiler script generation and Windows installer packaging."""

import subprocess
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from companion_packager.installer.inno_builder import (
    build_windows_installer,
    find_iscc,
    generate_inno_script,
)


def test_generate_inno_script_directives(tmp_path: Path):
    """Test Inno Setup script generation with required directives and sections."""
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "dummy.txt").write_text("dummy", encoding="utf-8")

    iss_path = tmp_path / "installer" / "AgentCompanionSetup.iss"
    result_path = generate_inno_script(
        bundle_dir,
        iss_path,
        app_name="Agent Companion",
        app_version="0.1.0",
    )

    assert result_path == iss_path
    assert iss_path.exists()

    content = iss_path.read_text(encoding="utf-8")

    # Assert definitions
    assert '#define SourceDir "' in content
    assert '#define AppName "Agent Companion"' in content
    assert '#define AppVersion "0.1.0"' in content

    # Assert [Setup] section
    assert "[Setup]" in content
    assert "AppName={#AppName}" in content
    assert "AppVersion={#AppVersion}" in content
    assert r"DefaultDirName={localappdata}\Programs\{#AppName}" in content
    assert "PrivilegesRequired=lowest" in content
    assert "OutputBaseFilename=AgentCompanion-Installer" in content

    # Assert [Files] section
    assert "[Files]" in content
    assert r'Source: "{#SourceDir}\*"; DestDir: "{app}";' in content

    # Assert [Icons] section
    assert "[Icons]" in content
    assert r'Name: "{autoprograms}\{#AppName}"; Filename: "{app}\start_companion.bat"' in content
    assert r'Name: "{autodesktop}\{#AppName}"; Filename: "{app}\start_companion.bat"' in content

    # Assert [Run] section
    assert "[Run]" in content
    assert r'Filename: "{app}\runtime\python.exe"' in content
    assert r'--find-links=""{app}\wheels""' in content
    assert "shared_contracts" in content
    assert "companion_packager" in content


def test_build_windows_installer_fallback_to_portable_zip(tmp_path: Path):
    """Test fallback when iscc.exe is absent: generates portable ZIP and setup scripts."""
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "agent_companion.env").write_text(
        'AGENT_COMPANION_DEVICE="cpu"\n', encoding="utf-8"
    )
    (bundle_dir / "start_companion.py").write_text("print('start')", encoding="utf-8")

    wheels_dir = bundle_dir / "wheels"
    wheels_dir.mkdir()
    (wheels_dir / "mock_pkg.whl").write_bytes(b"mock wheel")

    output_dir = tmp_path / "dist"

    with patch("companion_packager.installer.inno_builder.find_iscc", return_value=None):
        result_path = build_windows_installer(bundle_dir, output_dir)

    assert result_path == output_dir / "AgentCompanion-Windows-Portable.zip"
    assert result_path.exists()

    # Inspect zip contents
    with zipfile.ZipFile(result_path, "r") as zf:
        namelist = zf.namelist()
        assert "run_setup.bat" in namelist
        assert "install.bat" in namelist
        assert "start_companion.bat" in namelist
        assert "agent_companion.env" in namelist
        assert "wheels/mock_pkg.whl" in namelist or "wheels\\mock_pkg.whl" in namelist

        run_setup_content = zf.read("run_setup.bat").decode("utf-8")
        assert "pip install --no-index" in run_setup_content
        assert "%~dp0wheels" in run_setup_content
        assert "companion-packager" in run_setup_content or "companion_packager" in run_setup_content


def test_build_windows_installer_with_mock_iscc(tmp_path: Path):
    """Test invocation of ISCC.exe when compiler is present."""
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    output_dir = tmp_path / "dist"

    fake_iscc = tmp_path / "ISCC.exe"
    fake_iscc.write_text("mock iscc", encoding="utf-8")

    def mock_subprocess_run(cmd, capture_output=True, text=True):
        # Simulate iscc creating installer exe
        installer_exe = output_dir / "AgentCompanion-Installer.exe"
        installer_exe.write_bytes(b"MZ mock installer")
        return MagicMock(returncode=0, stdout="Compiled successfully", stderr="")

    with patch("companion_packager.installer.inno_builder.find_iscc", return_value=fake_iscc):
        with patch("subprocess.run", side_effect=mock_subprocess_run) as mock_run:
            result_path = build_windows_installer(bundle_dir, output_dir)

            assert mock_run.called
            assert result_path == output_dir / "AgentCompanion-Installer.exe"
            assert result_path.exists()
