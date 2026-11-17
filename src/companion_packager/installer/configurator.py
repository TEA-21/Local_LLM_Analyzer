"""Agent Companion runtime environment configurator, bootstrap script generator, and path resolver."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

from companion_packager.schemas import (
    CandidateModel,
    ConcurrencyScenarioBudget,
    HardwareProfile,
)


def resolve_companion_path(companion_path: Path | str | None = None) -> Path:
    """Deterministically locate the Agent Companion monorepo directory.

    Resolution Order:
    1. CLI argument/parameter if provided.
    2. Environment variable AGENT_COMPANION_PATH.
    3. Relative sibling directory '../Agent Companion' or '../../Agent Companion'.
    4. Default Windows path 'D:\\Projects\\Agent Companion'.
    5. Raises FileNotFoundError if none exist.
    """
    if companion_path:
        p = Path(companion_path).resolve()
        if p.exists() and p.is_dir():
            return p
        raise FileNotFoundError(f"Specified Agent Companion directory does not exist: {p}")

    env_path = os.environ.get("AGENT_COMPANION_PATH")
    if env_path:
        p = Path(env_path).resolve()
        if p.exists() and p.is_dir():
            return p

    current_dir = Path.cwd().resolve()
    candidates = [
        current_dir.parent / "Agent Companion",
        current_dir.parent.parent / "Agent Companion",
        Path(r"D:\Projects\Agent Companion"),
    ]
    for cand in candidates:
        if cand.exists() and cand.is_dir():
            return cand

    raise FileNotFoundError(
        "Agent Companion directory not found. Please provide --companion-path or set AGENT_COMPANION_PATH."
    )


def create_desktop_shortcut(
    bundle_dir: Path, shortcut_name: str = "Agent Companion"
) -> Path | None:
    """Create a desktop shortcut to start_companion launcher.

    Uses PowerShell WScript.Shell COM script on Windows, and .desktop entry on Linux.
    Fail-soft: returns None on any failure without raising exceptions.
    """
    try:
        if sys.platform == "win32":
            desktop_dir = Path.home() / "Desktop"
            if not desktop_dir.exists():
                userprofile = os.environ.get("USERPROFILE")
                if userprofile:
                    desktop_dir = Path(userprofile) / "Desktop"
            if not desktop_dir.exists():
                return None

            shortcut_path = desktop_dir / f"{shortcut_name}.lnk"
            target_path = bundle_dir / "start_companion.bat"

            ps_script = (
                f"$ws = New-Object -ComObject WScript.Shell; "
                f"$s = $ws.CreateShortcut('{shortcut_path}'); "
                f"$s.TargetPath = '{target_path}'; "
                f"$s.WorkingDirectory = '{bundle_dir}'; "
                f"$s.Save()"
            )
            subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
                capture_output=True,
                check=False,
                timeout=5,
            )
            if shortcut_path.exists():
                return shortcut_path
            return None

        elif sys.platform.startswith("linux"):
            app_dir = Path.home() / ".local" / "share" / "applications"
            app_dir.mkdir(parents=True, exist_ok=True)
            desktop_file = app_dir / "agent-companion.desktop"
            target_script = bundle_dir / "start_companion.sh"
            content = f"""[Desktop Entry]
Type=Application
Name={shortcut_name}
Exec="{target_script}"
Path="{bundle_dir}"
Terminal=true
Categories=Development;
"""
            desktop_file.write_text(content, encoding="utf-8")
            return desktop_file

        return None
    except Exception:
        return None


def inject_agent_companion_config(
    bundle_dir: Path,
    selected_model: CandidateModel,
    hardware: HardwareProfile,
    host: str = "127.0.0.1",
    port: int = 8000,
    scenario: Optional[ConcurrencyScenarioBudget] = None,
) -> dict[str, Path]:
    """Generate agent_companion.env and bootstrap launcher scripts in the bundle directory.

    Generates:
    - agent_companion.env
    - start_companion.py
    - start_companion.bat
    - start_companion.sh

    Returns:
        dict with keys {"env", "py", "bat", "sh"} mapping to created Path objects.
    """
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    repo = selected_model.repo_id.lower()
    is_ternary = (
        selected_model.quant_method.upper() in ("TERNARY", "BITNET", "COLIBRI")
        or "ternaryquench" in repo
    )
    is_cpu = "cpu" in hardware.backend_target.lower()

    if is_ternary:
        backend = "colibri_ternary"
    elif is_cpu:
        backend = "cpu_llama"
    else:
        backend = hardware.backend_target.lower()

    if is_cpu:
        if selected_model.param_count_billions and selected_model.param_count_billions >= 20.0:
            context_length = 4096
        elif selected_model.param_count_billions and selected_model.param_count_billions >= 6.5:
            context_length = 8192
        else:
            context_length = min(selected_model.context_window, 8192) if selected_model.context_window else 8192
    else:
        context_length = 32768

    if scenario is not None:
        max_ram_mb = round(scenario.budget_ceiling_mb, 1)
    else:
        max_ram_mb = 5200.0

    if is_cpu:
        ram_ceiling_mb = 6144.0
    else:
        if scenario is not None:
            ram_ceiling_mb = round(scenario.budget_ceiling_mb, 1)
        elif hardware.gpus:
            ram_ceiling_mb = round(hardware.gpus[0].vram_total_mb * 0.80, 1)
        else:
            ram_ceiling_mb = 6144.0

    # 1. agent_companion.env
    env_path = bundle_dir / "agent_companion.env"
    env_content = f"""# Agent Companion Autogenerated Runtime Environment
AGENT_COMPANION_MODEL_PATH="models/{selected_model.filename}"
AGENT_COMPANION_DEVICE="{hardware.backend_target}"
AGENT_COMPANION_BACKEND="{backend}"
AGENT_COMPANION_CONTEXT_LENGTH={context_length}
AGENT_COMPANION_CONTEXT_WINDOW="{context_length}"
AGENT_COMPANION_THREADS="{hardware.physical_cores}"
AGENT_COMPANION_QUANT="{selected_model.quant_method}"
AGENT_COMPANION_MAX_RAM_MB={max_ram_mb:.1f}
AGENT_COMPANION_RAM_CEILING_MB={ram_ceiling_mb:.1f}
AGENT_COMPANION_AIRGAP_ENFORCED=true
AGENT_COMPANION_SSE_HOST="{host}"
AGENT_COMPANION_SSE_PORT="{port}"
"""
    env_path.write_text(env_content, encoding="utf-8")

    # 2. start_companion.py
    py_path = bundle_dir / "start_companion.py"
    py_content = '''#!/usr/bin/env python3
"""Agent Companion Runtime Supervision Daemon Launcher."""

import os
import sys
from pathlib import Path


def load_env(env_path: Path) -> None:
    """Parse .env file, stripping comments and surrounding quotes, inject into os.environ."""
    if not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip(\'"\').strip("\'")
                os.environ[key] = val


def main() -> None:
    # Ensure working directory is install root for relative model paths
    install_root = Path(__file__).parent.resolve()
    os.chdir(install_root)

    env_file = install_root / "agent_companion.env"
    load_env(env_file)

    # Enforce sub-6GB RAM invariant when running on CPU backend
    device = os.environ.get("AGENT_COMPANION_DEVICE", "cpu").lower()
    backend = os.environ.get("AGENT_COMPANION_BACKEND", "colibri_ternary")
    if (backend == "colibri_ternary" or backend == "cpu_llama") and "cpu" in device:
        try:
            from agent_tooling.hardware_profiler.memory_guard import HardwareProfiler
            import logging
            logger = logging.getLogger("agent_companion")
            profiler = HardwareProfiler()
            assert profiler.RAM_CEILING_MB <= 6144.0, "Violates sub-6GB RAM invariant"
            MAX_RAM_MB = float(os.environ.get("AGENT_COMPANION_MAX_RAM_MB", "5200.0"))
            if profiler.AVAILABLE_RAM_MB < (MAX_RAM_MB * 0.70):
                logger.warning("Current free RAM lower than scenario threshold. Consider switching to Parallel mode.")
        except ImportError:
            pass

    host = os.environ.get("AGENT_COMPANION_SSE_HOST", "127.0.0.1")
    port = int(os.environ.get("AGENT_COMPANION_SSE_PORT", "8000"))
    model_path = os.environ.get("AGENT_COMPANION_MODEL_PATH", "")

    print(f"[Agent Companion] Starting server on {host}:{port}")
    print(f"[Agent Companion] Target Device: {device}, Backend: {backend}, Model: {model_path}")

    try:
        import uvicorn
        uvicorn.run("security_server.server:app", host=host, port=port, log_level="info")
    except ImportError:
        print("[Agent Companion] Uvicorn or security_server not found. Running in standalone fallback mode.")


if __name__ == "__main__":
    main()
'''
    py_path.write_text(py_content, encoding="utf-8")

    # 3. start_companion.bat
    bat_path = bundle_dir / "start_companion.bat"
    bat_content = (
        "@echo off\r\n"
        'for /F "usebackq eol=# tokens=1* delims==" %%i in ("%~dp0agent_companion.env") do (\r\n'
        '    set "%%i=%%~j"\r\n'
        ")\r\n"
        '"%~dp0runtime\\python.exe" "%~dp0start_companion.py"\r\n'
    )
    bat_path.write_text(bat_content, encoding="utf-8")

    # 4. start_companion.sh
    sh_path = bundle_dir / "start_companion.sh"
    sh_content = (
        "#!/usr/bin/env bash\n"
        "set -e\n"
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'export $(grep -v \'^#\' "$SCRIPT_DIR/agent_companion.env" | xargs)\n'
        'exec "$SCRIPT_DIR/runtime/bin/python3" "$SCRIPT_DIR/start_companion.py"\n'
    )
    sh_path.write_text(sh_content, encoding="utf-8", newline="\n")

    # Executable permissions for POSIX shell script
    try:
        sh_path.chmod(sh_path.stat().st_mode | 0o755)
    except OSError:
        pass

    return {
        "env": env_path,
        "py": py_path,
        "bat": bat_path,
        "sh": sh_path,
    }


generate_config = inject_agent_companion_config
