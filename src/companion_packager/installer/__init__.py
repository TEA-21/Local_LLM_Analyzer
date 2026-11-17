"""Installer subsystem: resumable downloader, configuration injector, wheel vendorizer, and packaging engine."""

from companion_packager.installer.configurator import (
    create_desktop_shortcut,
    generate_config,
    inject_agent_companion_config,
    resolve_companion_path,
)
from companion_packager.installer.downloader import (
    DiskSpaceError,
    DownloadChecksumError,
    DownloadError,
    clean_hash,
    compute_sha256,
    download_file,
)
from companion_packager.installer.inno_builder import (
    build_windows_installer,
    find_iscc,
    generate_inno_script,
)
from companion_packager.installer.sfx_builder import build_posix_sfx
from companion_packager.installer.wheel_vendor import (
    acquire_python_standalone,
    build_packager_wheel,
    extract_and_filter_requirements,
    stage_and_build_monorepo_wheels,
    vendor_third_party_wheels,
)

__all__ = [
    "download_file",
    "compute_sha256",
    "clean_hash",
    "DownloadError",
    "DownloadChecksumError",
    "DiskSpaceError",
    "inject_agent_companion_config",
    "generate_config",
    "create_desktop_shortcut",
    "resolve_companion_path",
    "stage_and_build_monorepo_wheels",
    "build_packager_wheel",
    "extract_and_filter_requirements",
    "acquire_python_standalone",
    "vendor_third_party_wheels",
    "find_iscc",
    "generate_inno_script",
    "build_windows_installer",
    "build_posix_sfx",
]
