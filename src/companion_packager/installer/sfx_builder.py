"""POSIX Self-Extracting Archive (.run) builder for Linux and macOS."""

import io
import os
import stat
import tarfile
from pathlib import Path


def build_posix_sfx(
    bundle_dir: Path,
    output_run_path: Path,
    app_name: str = "Agent Companion",
) -> Path:
    """
    Build a self-extracting shell archive (.run) for POSIX environments (Linux & macOS).
    Combines a bash extraction header with a compressed tar.gz bundle.
    """
    bundle_dir = Path(bundle_dir).resolve()
    output_run_path = Path(output_run_path).resolve()
    output_run_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure POSIX launcher scripts exist in bundle
    decompress_sh = bundle_dir / "decompress_and_run.sh"
    if not decompress_sh.exists():
        decompress_sh.write_text(
            "#!/usr/bin/env bash\n"
            "set -e\n"
            'DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
            'echo "[Agent Companion Setup] Installing offline runtime dependencies..."\n'
            '"$DIR/runtime/bin/python3" -m pip install --no-index --find-links="$DIR/wheels" shared_contracts model_runtime security_server document_pipeline agent_tooling companion_packager\n'
            '"$DIR/runtime/bin/python3" -m companion_packager.cli install --install-dir "$DIR" --non-interactive\n',
            encoding="utf-8",
        )
        try:
            decompress_sh.chmod(0o755)
        except Exception:
            pass

    start_sh = bundle_dir / "start_companion.sh"
    if not start_sh.exists():
        start_sh.write_text(
            "#!/usr/bin/env bash\n"
            'DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
            "set -a\n"
            '[ -f "$DIR/agent_companion.env" ] && source "$DIR/agent_companion.env"\n'
            "set +a\n"
            '"$DIR/runtime/bin/python3" "$DIR/start_companion.py"\n',
            encoding="utf-8",
        )
        try:
            start_sh.chmod(0o755)
        except Exception:
            pass

    # Create tar.gz archive in memory
    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w:gz") as tar:
        for item in sorted(bundle_dir.iterdir()):
            tar.add(str(item), arcname=item.name)

    tar_bytes = tar_buf.getvalue()

    header = (
        "#!/usr/bin/env bash\n"
        f"# Self-Extracting Installer for {app_name}\n"
        "set -e\n\n"
        f'APP_NAME="{app_name}"\n'
        'DEFAULT_INSTALL_DIR="$HOME/.local/share/agent-companion"\n'
        'INSTALL_DIR="$DEFAULT_INSTALL_DIR"\n\n'
        "while [[ $# -gt 0 ]]; do\n"
        "  case $1 in\n"
        "    --install-dir)\n"
        '      INSTALL_DIR="$2"\n'
        "      shift 2\n"
        "      ;;\n"
        "    *)\n"
        "      shift\n"
        "      ;;\n"
        "  esac\n"
        "done\n\n"
        'echo "[$APP_NAME] Installing to $INSTALL_DIR..."\n'
        'mkdir -p "$INSTALL_DIR"\n\n'
        "# Extract archive payload using sed\n"
        'sed -e \'1,/^__ARCHIVE_BELOW__/d\' "$0" | tar -xz -C "$INSTALL_DIR"\n\n'
        "# Offline pip install\n"
        'if [ -f "$INSTALL_DIR/runtime/bin/python3" ]; then\n'
        '  PYTHON_BIN="$INSTALL_DIR/runtime/bin/python3"\n'
        'elif [ -f "$INSTALL_DIR/runtime/python" ]; then\n'
        '  PYTHON_BIN="$INSTALL_DIR/runtime/python"\n'
        "else\n"
        '  PYTHON_BIN="python3"\n'
        "fi\n\n"
        'if [ -d "$INSTALL_DIR/wheels" ]; then\n'
        '  echo "[$APP_NAME] Installing offline dependencies..."\n'
        '  "$PYTHON_BIN" -m pip install --no-index --find-links="$INSTALL_DIR/wheels" shared_contracts model_runtime security_server document_pipeline agent_tooling companion_packager 2>/dev/null || true\n'
        "fi\n\n"
        'if [ -f "$INSTALL_DIR/decompress_and_run.sh" ]; then\n'
        '  chmod +x "$INSTALL_DIR/decompress_and_run.sh"\n'
        "fi\n"
        'if [ -f "$INSTALL_DIR/start_companion.sh" ]; then\n'
        '  chmod +x "$INSTALL_DIR/start_companion.sh"\n'
        "fi\n\n"
        "# Symlink launcher\n"
        'mkdir -p "$HOME/.local/bin"\n'
        'if [ -f "$INSTALL_DIR/start_companion.sh" ]; then\n'
        '  ln -sf "$INSTALL_DIR/start_companion.sh" "$HOME/.local/bin/agent-companion" 2>/dev/null || true\n'
        "fi\n\n"
        'echo "[$APP_NAME] Installation complete at $INSTALL_DIR."\n'
        "exit 0\n\n"
        "__ARCHIVE_BELOW__\n"
    )

    header_bytes = header.encode("utf-8")
    output_run_path.write_bytes(header_bytes + tar_bytes)

    try:
        current_mode = output_run_path.stat().st_mode
        output_run_path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except Exception:
        pass

    return output_run_path
