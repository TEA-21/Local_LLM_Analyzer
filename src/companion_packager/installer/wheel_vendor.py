"""Wheel vendorization and standalone Python runtime acquisition."""

import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from pathlib import Path

import httpx

WORKSPACE_PACKAGES = {
    "shared-contracts",
    "model-runtime",
    "security-server",
    "document-pipeline",
    "agent-tooling",
    "companion-packager",
    "shared_contracts",
    "model_runtime",
    "security_server",
    "document_pipeline",
    "agent_tooling",
    "companion_packager",
}

PACKAGER_DEPENDENCIES = [
    "typer>=0.12.0",
    "rich>=13.7.0",
    "pydantic>=2.7.0",
    "psutil>=5.9.8",
    "py-cpuinfo>=9.0.0",
    "huggingface-hub>=0.23.0",
    "httpx>=0.27.0",
    "build>=1.2.0",
    "hatchling>=1.21.0",
]

STANDALONE_PYTHON_URLS = {
    "windows": "https://github.com/astral-sh/python-build-standalone/releases/download/20241016/cpython-3.11.10+20241016-x86_64-pc-windows-msvc-shared-install_only.tar.gz",
    "linux": "https://github.com/astral-sh/python-build-standalone/releases/download/20241016/cpython-3.11.10+20241016-x86_64-unknown-linux-gnu-install_only.tar.gz",
    "macos": "https://github.com/astral-sh/python-build-standalone/releases/download/20241016/cpython-3.11.10+20241016-aarch64-apple-darwin-install_only.tar.gz",
    "darwin": "https://github.com/astral-sh/python-build-standalone/releases/download/20241016/cpython-3.11.10+20241016-aarch64-apple-darwin-install_only.tar.gz",
}

PLATFORM_TAGS = {
    "windows": "win_amd64",
    "win32": "win_amd64",
    "linux": "manylinux2014_x86_64",
    "macos": "macosx_11_0_arm64",
    "darwin": "macosx_11_0_arm64",
}


def stage_and_build_monorepo_wheels(
    companion_path: Path,
    output_wheels_dir: Path,
) -> list[Path]:
    """
    Stage monorepo packages in isolated temporary directories and build wheels with hatchling.
    Leaves the source repository untouched (injecting stub README.md strictly in staging).
    """
    companion_path = Path(companion_path).resolve()
    output_wheels_dir = Path(output_wheels_dir).resolve()
    output_wheels_dir.mkdir(parents=True, exist_ok=True)

    packages_dir = companion_path / "packages"
    if not packages_dir.exists():
        raise FileNotFoundError(f"Packages directory not found at {packages_dir}")

    target_packages = [
        "shared_contracts",
        "model_runtime",
        "security_server",
        "document_pipeline",
        "agent_tooling",
    ]

    built_wheels: list[Path] = []
    initial_wheels = set(output_wheels_dir.glob("*.whl"))

    for pkg_name in target_packages:
        pkg_path = packages_dir / pkg_name
        if not pkg_path.exists():
            hyphen_path = packages_dir / pkg_name.replace("_", "-")
            if hyphen_path.exists():
                pkg_path = hyphen_path
            else:
                continue

        if not (pkg_path / "pyproject.toml").exists():
            continue

        with tempfile.TemporaryDirectory() as tmp_dir:
            staged_pkg = Path(tmp_dir) / pkg_name
            shutil.copytree(pkg_path, staged_pkg)

            # Check for README.md; inject empty stub if missing (avoids hatchling build failure)
            readme_path = staged_pkg / "README.md"
            if not readme_path.exists():
                readme_path.write_text(f"# {pkg_name}\n", encoding="utf-8")

            # Run python -m build --wheel --no-isolation
            cmd = [
                sys.executable,
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                str(output_wheels_dir),
            ]
            result = subprocess.run(
                cmd,
                cwd=str(staged_pkg),
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Failed to build wheel for {pkg_name}: {result.stderr or result.stdout}"
                )

    current_wheels = set(output_wheels_dir.glob("*.whl"))
    new_wheels = current_wheels - initial_wheels
    if new_wheels:
        built_wheels = sorted(new_wheels)
    else:
        built_wheels = sorted(current_wheels)

    return built_wheels


def build_packager_wheel(
    output_wheels_dir: Path,
    packager_root: Path | None = None,
) -> Path:
    """Build companion-packager itself into a wheel and place it in output_wheels_dir."""
    output_wheels_dir = Path(output_wheels_dir).resolve()
    output_wheels_dir.mkdir(parents=True, exist_ok=True)

    if packager_root is None:
        packager_root = Path(__file__).resolve().parents[3]

    packager_root = Path(packager_root).resolve()
    cmd = [
        sys.executable,
        "-m",
        "build",
        "--wheel",
        "--no-isolation",
        "--outdir",
        str(output_wheels_dir),
    ]
    result = subprocess.run(
        cmd,
        cwd=str(packager_root),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to build companion-packager wheel: {result.stderr or result.stdout}"
        )

    candidates = list(output_wheels_dir.glob("companion_packager-*.whl"))
    if not candidates:
        raise FileNotFoundError("companion-packager wheel was not produced")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def extract_and_filter_requirements(companion_path: Path) -> list[str]:
    """
    Parse [project].dependencies from each package pyproject.toml in companion_path.
    Strips workspace packages and appends companion-packager's dependencies.
    Returns sorted unique filtered requirements.
    """
    companion_path = Path(companion_path).resolve()
    packages_dir = companion_path / "packages"

    filtered_reqs: set[str] = set()

    if packages_dir.exists() and packages_dir.is_dir():
        for pkg_dir in packages_dir.iterdir():
            if not pkg_dir.is_dir():
                continue
            pyproject_file = pkg_dir / "pyproject.toml"
            if not pyproject_file.exists():
                continue

            with open(pyproject_file, "rb") as f:
                data = tomllib.load(f)
                deps = data.get("project", {}).get("dependencies", [])
                for dep in deps:
                    base_name = (
                        re.split(r"[<>=!~; ]", dep)[0]
                        .strip()
                        .lower()
                        .replace("_", "-")
                    )
                    if (
                        base_name in WORKSPACE_PACKAGES
                        or base_name.replace("-", "_") in WORKSPACE_PACKAGES
                    ):
                        continue
                    filtered_reqs.add(dep)

    # Append companion-packager dependencies
    for dep in PACKAGER_DEPENDENCIES:
        filtered_reqs.add(dep)

    return sorted(list(filtered_reqs))


def acquire_python_standalone(
    target_os: str,
    target_dir: Path,
    python_dist_path: Path | None = None,
) -> Path:
    """
    Acquire and extract pinned python-build-standalone archive (release 20241016).
    Strips the leading root directory (e.g. 'python/') during extraction so that binaries
    reside directly at <target_dir>/python.exe (Windows) or <target_dir>/bin/python3 (POSIX).
    Returns the Path to the python executable.
    """
    target_dir = Path(target_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    norm_os = target_os.lower().strip()
    if norm_os in ("windows", "win32", "win"):
        os_key = "windows"
        relative_bin = Path("python.exe")
    elif norm_os in ("macos", "darwin", "mac", "osx"):
        os_key = "macos"
        relative_bin = Path("bin") / "python3"
    else:
        os_key = "linux"
        relative_bin = Path("bin") / "python3"

    if python_dist_path and Path(python_dist_path).exists():
        archive_path = Path(python_dist_path).resolve()
    else:
        url = STANDALONE_PYTHON_URLS.get(os_key, STANDALONE_PYTHON_URLS["windows"])
        cache_dir = Path.home() / ".cache" / "companion-packager" / "python"
        cache_dir.mkdir(parents=True, exist_ok=True)
        filename = url.rsplit("/", 1)[-1]
        archive_path = cache_dir / filename

        if not archive_path.exists():
            with httpx.Client(follow_redirects=True, timeout=60.0) as client:
                resp = client.get(url)
                resp.raise_for_status()
                part_path = archive_path.with_suffix(".part")
                part_path.write_bytes(resp.content)
                part_path.replace(archive_path)

    # Extract stripping root directory
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            parts = Path(member.name).parts
            if len(parts) <= 1:
                continue
            # Strip top-level directory
            member.name = "/".join(parts[1:])
            try:
                tar.extract(member, path=target_dir, filter="data")
            except (TypeError, AttributeError):
                tar.extract(member, path=target_dir)

    python_bin = target_dir / relative_bin
    if python_bin.exists():
        return python_bin
    # Defensive fallbacks for cross-platform mocks
    if (target_dir / "python.exe").exists():
        return target_dir / "python.exe"
    if (target_dir / "bin" / "python3").exists():
        return target_dir / "bin" / "python3"
    if (target_dir / "bin" / "python").exists():
        return target_dir / "bin" / "python"
    return python_bin


def vendor_third_party_wheels(
    requirements: list[str],
    output_wheels_dir: Path,
    target_os: str = "windows",
    wheels_cache_dir: Path | None = None,
    offline: bool = False,
) -> list[Path]:
    """
    Vendor third-party wheels for requirements.
    In offline mode, copies pre-cached wheels from wheels_cache_dir or default cache;
    raises ValueError if cache is empty.
    In online mode, invokes pip download with target platform and cp311 ABI.
    Returns list of wheel paths in output_wheels_dir.
    """
    output_wheels_dir = Path(output_wheels_dir).resolve()
    output_wheels_dir.mkdir(parents=True, exist_ok=True)

    default_cache_dir = Path.home() / ".cache" / "companion-packager" / "wheels"
    copied_from_cache = False

    search_caches = []
    if wheels_cache_dir is not None:
        search_caches.append(Path(wheels_cache_dir).resolve())
    search_caches.append(default_cache_dir)

    for cache in search_caches:
        if cache.exists() and cache.is_dir():
            cached_wheels = list(cache.glob("*.whl"))
            if cached_wheels:
                for whl in cached_wheels:
                    shutil.copy2(whl, output_wheels_dir / whl.name)
                copied_from_cache = True
                break

    if offline:
        if not copied_from_cache and not list(output_wheels_dir.glob("*.whl")):
            raise ValueError("Offline mode requested but no wheels cache found")
        return sorted(output_wheels_dir.glob("*.whl"))

    if requirements:
        norm_os = target_os.lower().strip()
        platform_tag = PLATFORM_TAGS.get(norm_os, "win_amd64")

        with tempfile.NamedTemporaryFile(
            "w", suffix="_reqs.txt", delete=False, encoding="utf-8"
        ) as req_file:
            req_file.write("\n".join(requirements) + "\n")
            req_path = Path(req_file.name)

        try:
            cmd = [
                sys.executable,
                "-m",
                "pip",
                "download",
                "--dest",
                str(output_wheels_dir),
                "--python-version",
                "3.11",
                "--platform",
                platform_tag,
                "--abi",
                "cp311",
                "--only-binary=:all:",
                "--implementation",
                "cp",
                "-r",
                str(req_path),
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Failed to vendor third-party wheels: {result.stderr or result.stdout}"
                )
        finally:
            if req_path.exists():
                req_path.unlink()

    return sorted(output_wheels_dir.glob("*.whl"))
