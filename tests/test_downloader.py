"""Unit tests for resumable model downloader (HTTP 200, 206 resume, 416 range, SHA-256 integrity, disk space)."""

import hashlib
import re
from pathlib import Path

import httpx
import pytest

from companion_packager.installer.downloader import (
    DiskSpaceError,
    DownloadChecksumError,
    DownloadError,
    clean_hash,
    compute_sha256,
    download_file,
)


@pytest.fixture
def test_data():
    # 2048 bytes of deterministic test data
    return b"Agent-Companion-Model-Weights-Block-Verification-Payload-1234567890\n" * 32


@pytest.fixture
def test_sha256(test_data):
    return hashlib.sha256(test_data).hexdigest().lower()


def test_clean_hash():
    """Verify normalization of ETags and hex hashes."""
    assert clean_hash('W/"a1b2c3"') == "a1b2c3"
    assert clean_hash('w/"A1B2C3"') == "a1b2c3"
    assert clean_hash('  "abcdef0123456789"  ') == "abcdef0123456789"
    assert clean_hash("ABCDEF") == "abcdef"
    assert clean_hash(None) is None
    assert clean_hash("") is None


def test_compute_sha256(tmp_path, test_data, test_sha256):
    """Verify streaming SHA-256 calculation."""
    file_path = tmp_path / "sample.bin"
    file_path.write_bytes(test_data)
    calculated = compute_sha256(file_path, chunk_size=256)
    assert calculated == test_sha256


def test_download_file_200_full_download(tmp_path, test_data, test_sha256):
    """Test standard HTTP 200 download from byte 0 to completion."""
    dest_path = tmp_path / "model.gguf"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "HEAD":
            return httpx.Response(200, headers={"Content-Length": str(len(test_data))})
        return httpx.Response(200, content=test_data, headers={"Content-Length": str(len(test_data))})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = download_file(
        url="https://mock.repo/model.gguf",
        dest_path=dest_path,
        expected_sha256=test_sha256,
        chunk_size=512,
        show_progress=False,
        client=client,
    )

    assert result == dest_path
    assert dest_path.exists()
    assert dest_path.read_bytes() == test_data
    # Staging .part file must be cleaned up
    part_path = dest_path.with_name(dest_path.name + ".part")
    assert not part_path.exists()


def test_download_file_206_resume_partial(tmp_path, test_data, test_sha256):
    """Test HTTP 206 partial resume when .part file already contains initial bytes."""
    dest_path = tmp_path / "model.gguf"
    part_path = dest_path.with_name(dest_path.name + ".part")

    # Pre-write 500 bytes into .part
    split_idx = 500
    part_path.write_bytes(test_data[:split_idx])

    range_requested = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "HEAD":
            return httpx.Response(200, headers={"Content-Length": str(len(test_data))})

        range_hdr = request.headers.get("Range")
        range_requested.append(range_hdr)

        if range_hdr == f"bytes={split_idx}-":
            remaining = test_data[split_idx:]
            return httpx.Response(
                206,
                content=remaining,
                headers={
                    "Content-Range": f"bytes {split_idx}-{len(test_data)-1}/{len(test_data)}",
                    "Content-Length": str(len(remaining)),
                },
            )
        return httpx.Response(200, content=test_data, headers={"Content-Length": str(len(test_data))})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = download_file(
        url="https://mock.repo/model.gguf",
        dest_path=dest_path,
        expected_sha256=test_sha256,
        chunk_size=256,
        show_progress=False,
        client=client,
    )

    assert range_requested == [f"bytes={split_idx}-"]
    assert result == dest_path
    assert dest_path.exists()
    assert dest_path.read_bytes() == test_data
    assert not part_path.exists()


def test_download_file_200_server_ignores_range(tmp_path, test_data, test_sha256):
    """Test server returning HTTP 200 instead of 206 (ignoring range) resets .part to byte 0."""
    dest_path = tmp_path / "model.gguf"
    part_path = dest_path.with_name(dest_path.name + ".part")

    # Staged partial file with old corrupted content
    part_path.write_bytes(b"corrupt-data-prefix" * 10)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=test_data, headers={"Content-Length": str(len(test_data))})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = download_file(
        url="https://mock.repo/model.gguf",
        dest_path=dest_path,
        expected_sha256=test_sha256,
        show_progress=False,
        client=client,
    )

    assert result == dest_path
    assert dest_path.read_bytes() == test_data
    assert not part_path.exists()


def test_download_file_416_already_fully_downloaded(tmp_path, test_data, test_sha256):
    """Test HTTP 416 when .part is already fully downloaded leads to successful SHA verification and rename."""
    dest_path = tmp_path / "model.gguf"
    part_path = dest_path.with_name(dest_path.name + ".part")

    # Entire file already staged in .part
    part_path.write_bytes(test_data)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            416,
            headers={"Content-Range": f"bytes */{len(test_data)}"},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = download_file(
        url="https://mock.repo/model.gguf",
        dest_path=dest_path,
        expected_sha256=test_sha256,
        expected_size=len(test_data),
        show_progress=False,
        client=client,
    )

    assert result == dest_path
    assert dest_path.exists()
    assert dest_path.read_bytes() == test_data
    assert not part_path.exists()


def test_download_file_416_corrupt_size_restarts_stream(tmp_path, test_data, test_sha256):
    """Test HTTP 416 with bad/mismatched .part size truncates and re-requests full body."""
    dest_path = tmp_path / "model.gguf"
    part_path = dest_path.with_name(dest_path.name + ".part")

    # Oversized or corrupt part file
    part_path.write_bytes(b"junk" * 1000)

    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        if "Range" in request.headers:
            return httpx.Response(
                416,
                headers={"Content-Range": f"bytes */{len(test_data)}"},
            )
        return httpx.Response(200, content=test_data, headers={"Content-Length": str(len(test_data))})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = download_file(
        url="https://mock.repo/model.gguf",
        dest_path=dest_path,
        expected_sha256=test_sha256,
        show_progress=False,
        client=client,
    )

    assert request_count >= 2
    assert result == dest_path
    assert dest_path.read_bytes() == test_data


def test_download_file_checksum_mismatch(tmp_path, test_data):
    """Test SHA-256 checksum mismatch renames .part to .corrupt and raises DownloadChecksumError."""
    dest_path = tmp_path / "model.gguf"
    part_path = dest_path.with_name(dest_path.name + ".part")
    corrupt_path = dest_path.with_name(dest_path.name + ".corrupt")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=test_data, headers={"Content-Length": str(len(test_data))})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    bad_sha = "0000000000000000000000000000000000000000000000000000000000000000"

    with pytest.raises(DownloadChecksumError) as exc_info:
        download_file(
            url="https://mock.repo/model.gguf",
            dest_path=dest_path,
            expected_sha256=bad_sha,
            show_progress=False,
            client=client,
        )

    assert "Checksum mismatch" in str(exc_info.value)
    assert not dest_path.exists()
    assert not part_path.exists()
    assert corrupt_path.exists()
    assert corrupt_path.read_bytes() == test_data


def test_download_file_idempotency_skip(tmp_path, test_data, test_sha256):
    """Test that existing valid destination file skips HTTP requests entirely."""
    dest_path = tmp_path / "model.gguf"
    dest_path.write_bytes(test_data)

    http_called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal http_called
        http_called = True
        return httpx.Response(500)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = download_file(
        url="https://mock.repo/model.gguf",
        dest_path=dest_path,
        expected_sha256=test_sha256,
        client=client,
    )

    assert result == dest_path
    assert not http_called


def test_download_file_disk_space_error(tmp_path, monkeypatch):
    """Test insufficient disk space raises DiskSpaceError without initiating download."""
    dest_path = tmp_path / "model.gguf"

    # Simulate insufficient disk space
    monkeypatch.setattr(
        "companion_packager.installer.downloader.check_disk_space_preflight",
        lambda *args, **kwargs: (False, 0.5, 10.0),
    )

    client = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200)))

    with pytest.raises(DiskSpaceError) as exc_info:
        download_file(
            url="https://mock.repo/model.gguf",
            dest_path=dest_path,
            expected_size=10 * 1024 * 1024,
            client=client,
        )

    assert "Insufficient disk space" in str(exc_info.value)
    assert not dest_path.exists()


def test_download_file_http_error(tmp_path):
    """Test HTTP error status codes (e.g. 404) raise DownloadError."""
    dest_path = tmp_path / "model.gguf"

    client = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(404)))

    with pytest.raises(DownloadError) as exc_info:
        download_file(
            url="https://mock.repo/nonexistent.gguf",
            dest_path=dest_path,
            client=client,
        )

    assert "HTTP error 404" in str(exc_info.value)
    assert not dest_path.exists()
