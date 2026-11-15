"""Installer subsystem: resumable downloader, configuration injector, and path resolver."""

from companion_packager.installer.downloader import (
    DiskSpaceError,
    DownloadChecksumError,
    DownloadError,
    clean_hash,
    compute_sha256,
    download_file,
)
from companion_packager.installer.configurator import (
    create_desktop_shortcut,
    inject_agent_companion_config,
    resolve_companion_path,
)

__all__ = [
    "download_file",
    "compute_sha256",
    "clean_hash",
    "DownloadError",
    "DownloadChecksumError",
    "DiskSpaceError",
    "inject_agent_companion_config",
    "create_desktop_shortcut",
    "resolve_companion_path",
]
