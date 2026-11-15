"""Resumable, chunked model downloader with HTTP range support, SHA-256 integrity verification, and pre-flight disk checks."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Optional

import httpx
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from companion_packager.hardware.detector import check_disk_space_preflight


class DownloadError(Exception):
    """Base exception for all download errors."""
    pass


class DownloadChecksumError(DownloadError):
    """Raised when downloaded file SHA-256 does not match expected checksum."""
    pass


class DiskSpaceError(DownloadError):
    """Raised when host disk space is insufficient for download and staging."""
    pass


def clean_hash(hash_str: str | None) -> str | None:
    """Normalize and clean SHA-256 hash or ETag string.

    Strips weak ETag prefixes (W/), surrounding quotes, and whitespace,
    and returns lowercase hex string.
    """
    if not hash_str:
        return None
    h = hash_str.strip()
    if h.startswith("W/") or h.startswith("w/"):
        h = h[2:]
    h = h.strip('"\'').strip()
    return h.lower()


def compute_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 hash of a file using streaming chunks."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def _stream_to_file(
    response: httpx.Response,
    file_path: Path,
    mode: str,
    show_progress: bool,
    total_size: int | None,
    initial_bytes: int,
    chunk_size: int,
) -> None:
    """Stream response body chunks into target file with optional Rich progress bar."""
    with open(file_path, mode) as f:
        if show_progress:
            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.1f}%",
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                transient=False,
            ) as progress:
                task_id = progress.add_task(
                    file_path.name,
                    total=total_size,
                    completed=initial_bytes,
                )
                for chunk in response.iter_bytes(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        progress.update(task_id, advance=len(chunk))
        else:
            for chunk in response.iter_bytes(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)


def download_file(
    url: str,
    dest_path: Path,
    expected_sha256: str | None = None,
    chunk_size: int = 1024 * 1024,
    show_progress: bool = True,
    expected_size: int | None = None,
    client: httpx.Client | None = None,
) -> Path:
    """Download a file with HTTP range resume, atomic staging, and SHA-256 integrity check.

    Args:
        url: Remote download URL.
        dest_path: Final destination path for the verified file.
        expected_sha256: Optional expected SHA-256 hash for integrity validation.
        chunk_size: Stream buffer size in bytes (default 1 MB).
        show_progress: Whether to render Rich progress bar during download.
        expected_size: Optional expected file size in bytes for pre-flight disk check.
        client: Optional httpx.Client instance for testing / reuse.

    Returns:
        Path to the verified destination file.

    Raises:
        DiskSpaceError: If insufficient disk space is available.
        DownloadChecksumError: If computed hash does not match expected_sha256.
        DownloadError: If HTTP or network errors occur.
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Idempotency Check on Destination File
    if dest_path.exists() and expected_sha256 is not None:
        cleaned_expected = clean_hash(expected_sha256)
        actual_sha = compute_sha256(dest_path, chunk_size=chunk_size)
        if actual_sha == cleaned_expected:
            return dest_path

    # HTTP Client Configuration
    own_client = client is None
    http_client = client or httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(30.0, read=60.0),
    )

    try:
        # 2. Pre-flight Disk Space Check
        estimated_size = expected_size or 0
        if estimated_size == 0:
            try:
                head_resp = http_client.head(url)
                if head_resp.status_code < 400 and "content-length" in head_resp.headers:
                    estimated_size = int(head_resp.headers["content-length"])
            except Exception:
                pass

        has_space, free_gb, req_gb = check_disk_space_preflight(
            dest_path.parent, required_model_bytes=estimated_size, is_installer_build=False
        )
        if not has_space:
            raise DiskSpaceError(
                f"Insufficient disk space in {dest_path.parent}: {free_gb} GB free, {req_gb} GB required."
            )

        # 3. .part File and Range Resume Setup
        part_path = dest_path.with_name(dest_path.name + ".part")
        existing_bytes = part_path.stat().st_size if part_path.exists() else 0

        headers: dict[str, str] = {}
        if existing_bytes > 0:
            headers["Range"] = f"bytes={existing_bytes}-"

        # 4. Execute Streaming Request
        with http_client.stream("GET", url, headers=headers) as response:
            if response.status_code == 416:
                # Range Not Satisfiable: existing .part may already be complete
                content_range = response.headers.get("Content-Range", "")
                match = re.search(r"/(\d+)$", content_range)
                server_total = int(match.group(1)) if match else None

                is_complete = False
                if server_total is not None and existing_bytes == server_total:
                    is_complete = True
                elif expected_size is not None and existing_bytes == expected_size:
                    is_complete = True
                elif expected_sha256 is not None and part_path.exists():
                    if compute_sha256(part_path, chunk_size=chunk_size) == clean_hash(expected_sha256):
                        is_complete = True

                if not is_complete:
                    # Truncate / delete partial file and restart full download
                    part_path.unlink(missing_ok=True)
                    existing_bytes = 0
                    with http_client.stream("GET", url) as full_resp:
                        full_resp.raise_for_status()
                        tot = int(full_resp.headers.get("Content-Length", expected_size or 0)) or None
                        _stream_to_file(
                            full_resp,
                            part_path,
                            "wb",
                            show_progress,
                            tot,
                            0,
                            chunk_size,
                        )
            elif response.status_code == 206:
                # Partial Content: server accepted Range header, append to .part
                content_range = response.headers.get("Content-Range", "")
                match = re.search(r"/(\d+)$", content_range)
                if match:
                    total_file_size = int(match.group(1))
                elif "Content-Length" in response.headers:
                    total_file_size = existing_bytes + int(response.headers["Content-Length"])
                else:
                    total_file_size = expected_size

                _stream_to_file(
                    response,
                    part_path,
                    "ab",
                    show_progress,
                    total_file_size,
                    existing_bytes,
                    chunk_size,
                )
            elif response.status_code == 200:
                # 200 OK: Range header ignored or not sent, stream full file from byte 0
                total_file_size = int(response.headers.get("Content-Length", expected_size or 0)) or None
                _stream_to_file(
                    response,
                    part_path,
                    "wb",
                    show_progress,
                    total_file_size,
                    0,
                    chunk_size,
                )
            else:
                response.raise_for_status()

        # 5. Integrity Verification
        if not part_path.exists():
            raise DownloadError(f"Staged part file not found after download: {part_path}")

        actual_sha256 = compute_sha256(part_path, chunk_size=chunk_size)
        if expected_sha256 is not None:
            cleaned_expected = clean_hash(expected_sha256)
            if actual_sha256 != cleaned_expected:
                corrupt_path = dest_path.with_name(dest_path.name + ".corrupt")
                if corrupt_path.exists():
                    corrupt_path.unlink(missing_ok=True)
                part_path.replace(corrupt_path)
                raise DownloadChecksumError(
                    f"Checksum mismatch for {dest_path.name}: expected {cleaned_expected}, got {actual_sha256}"
                )

        # 6. Atomic Finalization
        part_path.replace(dest_path)
        return dest_path

    except (DiskSpaceError, DownloadChecksumError):
        raise
    except httpx.HTTPStatusError as e:
        raise DownloadError(f"HTTP error {e.response.status_code} downloading {url}: {e}") from e
    except httpx.RequestError as e:
        raise DownloadError(f"Network error downloading {url}: {e}") from e
    except Exception as e:
        raise DownloadError(f"Failed to download {url}: {e}") from e
    finally:
        if own_client:
            http_client.close()
