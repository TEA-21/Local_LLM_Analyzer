"""Dynamic active process audit and memory categorization subsystem."""

from typing import Any, Dict, List, Set
import psutil

from companion_packager.schemas import ProcessCategory, ProcessMemorySummary

BROWSERS_SET: Set[str] = {
    "chrome",
    "chrome_proxy",
    "google chrome",
    "msedge",
    "brave",
    "firefox",
    "arc",
    "opera",
    "safari",
    "vivaldi",
    "tor",
}

IDES_SET: Set[str] = {
    "code",
    "antigravity",
    "antigravity ide",
    "idea64",
    "idea",
    "pycharm64",
    "pycharm",
    "cursor",
    "clion64",
    "clion",
    "sublime_text",
    "notepad++",
    "devenv",
    "webstorm64",
    "webstorm",
    "rider64",
    "goland64",
}

DEV_RUNTIMES_SET: Set[str] = {
    "node",
    "python",
    "python3",
    "pythonw",
    "docker",
    "dockerd",
    "wsl",
    "wslhost",
    "cargo",
    "rustc",
    "java",
    "javaw",
    "go",
    "git",
}

SYSTEM_SET: Set[str] = {
    "system",
    "registry",
    "smss",
    "csrss",
    "wininit",
    "services",
    "lsass",
    "svchost",
    "explorer",
    "init",
    "systemd",
    "launchd",
    "kernel_task",
}


def audit_active_processes() -> ProcessMemorySummary:
    """Enumerate running processes, categorize memory usage, and report top consumers."""
    browsers_mb = 0.0
    ides_mb = 0.0
    dev_runtimes_mb = 0.0
    system_mb = 0.0
    other_mb = 0.0
    total_scanned_mb = 0.0
    scanned_processes: List[Dict[str, Any]] = []

    for p in psutil.process_iter(["pid", "name", "memory_info"]):
        try:
            info = p.info if hasattr(p, "info") and isinstance(p.info, dict) else {}
            mem_info = info.get("memory_info")
            if mem_info is None or not hasattr(mem_info, "rss"):
                continue

            raw_name = info.get("name") or ""
            proc_name = raw_name.lower().removesuffix(".exe")
            tokens = set(proc_name.replace("-", " ").replace("_", " ").split())

            if proc_name in BROWSERS_SET or bool(tokens & BROWSERS_SET):
                category = ProcessCategory.BROWSERS
            elif proc_name in IDES_SET or bool(tokens & IDES_SET):
                category = ProcessCategory.IDES
            elif proc_name in DEV_RUNTIMES_SET or bool(tokens & DEV_RUNTIMES_SET):
                category = ProcessCategory.DEV_RUNTIMES
            elif proc_name in SYSTEM_SET or bool(tokens & SYSTEM_SET):
                category = ProcessCategory.SYSTEM
            else:
                category = ProcessCategory.OTHER

            rss_mb = round(mem_info.rss / (1024 * 1024), 1)

            if category == ProcessCategory.BROWSERS:
                browsers_mb += rss_mb
            elif category == ProcessCategory.IDES:
                ides_mb += rss_mb
            elif category == ProcessCategory.DEV_RUNTIMES:
                dev_runtimes_mb += rss_mb
            elif category == ProcessCategory.SYSTEM:
                system_mb += rss_mb
            else:
                other_mb += rss_mb

            total_scanned_mb += rss_mb

            pid_val = info.get("pid")
            pid = pid_val if isinstance(pid_val, int) else 0

            scanned_processes.append(
                {
                    "name": proc_name,
                    "rss_mb": rss_mb,
                    "category": category.value,
                    "pid": pid,
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    scanned_processes.sort(key=lambda x: x["rss_mb"], reverse=True)
    top_processes = scanned_processes[:5]

    return ProcessMemorySummary(
        browsers_mb=round(browsers_mb, 1),
        ides_mb=round(ides_mb, 1),
        dev_runtimes_mb=round(dev_runtimes_mb, 1),
        system_mb=round(system_mb, 1),
        other_mb=round(other_mb, 1),
        total_scanned_mb=round(total_scanned_mb, 1),
        top_processes=top_processes,
    )
