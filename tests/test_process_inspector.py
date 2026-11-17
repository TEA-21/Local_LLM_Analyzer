"""Tests for process inspection subsystem, categorization, and domain schemas."""

from collections import namedtuple
from unittest.mock import MagicMock, patch
import psutil
import pytest

from companion_packager.hardware.process_inspector import audit_active_processes
from companion_packager.schemas import (
    ConcurrencyScenario,
    ConcurrencyScenarioBudget,
    ProcessCategory,
    ProcessMemorySummary,
)
import companion_packager.models.schemas as models_schemas

MemInfo = namedtuple("MemInfo", ["rss"])


class MockProcess:
    """Mock psutil.Process yielding preconfigured info."""

    def __init__(self, pid: int, name: str, rss_bytes: int, raise_exc=None):
        self._raise_exc = raise_exc
        self._info = {
            "pid": pid,
            "name": name,
            "memory_info": MemInfo(rss=rss_bytes) if rss_bytes is not None else None,
        }

    @property
    def info(self):
        if self._raise_exc:
            raise self._raise_exc
        return self._info


def test_schema_definitions_and_reexport():
    """Verify ProcessCategory, ProcessMemorySummary, and ConcurrencyScenario schemas."""
    # Enum tests
    assert ProcessCategory.BROWSERS == "browsers"
    assert ProcessCategory.IDES == "ides"
    assert ProcessCategory.DEV_RUNTIMES == "dev_runtimes"
    assert ProcessCategory.SYSTEM == "system"
    assert ProcessCategory.OTHER == "other"

    assert ConcurrencyScenario.PARALLEL == "parallel"
    assert ConcurrencyScenario.MINIMAL_INTERFERENCE == "minimal_interference"
    assert ConcurrencyScenario.DEDICATED == "dedicated"

    # ProcessMemorySummary validation
    summary = ProcessMemorySummary(
        browsers_mb=1200.5,
        ides_mb=850.0,
        dev_runtimes_mb=420.2,
        system_mb=600.0,
        other_mb=310.1,
        total_scanned_mb=3380.8,
        top_processes=[
            {"name": "chrome", "rss_mb": 1200.5, "category": "browsers", "pid": 1001}
        ],
    )
    assert summary.browsers_mb == 1200.5
    assert len(summary.top_processes) == 1
    assert summary.top_processes[0]["name"] == "chrome"

    # ConcurrencyScenarioBudget validation
    budget = ConcurrencyScenarioBudget(
        scenario=ConcurrencyScenario.PARALLEL,
        title="Parallel Coexistence",
        description="Run alongside active applications",
        available_ram_mb=8000.0,
        reclaimable_ram_mb=0.0,
        budget_ceiling_mb=6000.0,
    )
    assert budget.scenario == ConcurrencyScenario.PARALLEL
    assert budget.recommended_scorecard is None

    # Verify models.schemas re-export shim
    assert models_schemas.ProcessCategory is ProcessCategory
    assert models_schemas.ProcessMemorySummary is ProcessMemorySummary
    assert models_schemas.ConcurrencyScenario is ConcurrencyScenario
    assert models_schemas.ConcurrencyScenarioBudget is ConcurrencyScenarioBudget


def test_process_inspector_categorization_and_exe_stripping():
    """Verify that process names are correctly normalized (.exe removed) and categorized."""
    mock_procs = [
        # Browsers (.exe and bare)
        MockProcess(101, "chrome.exe", 100 * 1024 * 1024),
        MockProcess(102, "msedge.exe", 80 * 1024 * 1024),
        MockProcess(103, "firefox", 70 * 1024 * 1024),
        MockProcess(104, "google-chrome", 60 * 1024 * 1024),
        # IDEs
        MockProcess(201, "Code.exe", 200 * 1024 * 1024),
        MockProcess(202, "cursor", 150 * 1024 * 1024),
        MockProcess(203, "idea64.exe", 180 * 1024 * 1024),
        MockProcess(204, "antigravity_ide.exe", 90 * 1024 * 1024),
        # Dev Runtimes
        MockProcess(301, "node.exe", 110 * 1024 * 1024),
        MockProcess(302, "python3", 85 * 1024 * 1024),
        MockProcess(303, "docker.exe", 130 * 1024 * 1024),
        # System
        MockProcess(401, "svchost.exe", 95 * 1024 * 1024),
        MockProcess(402, "explorer.exe", 75 * 1024 * 1024),
        MockProcess(403, "systemd", 30 * 1024 * 1024),
        # Other
        MockProcess(501, "spotify.exe", 120 * 1024 * 1024),
        MockProcess(502, "slack", 140 * 1024 * 1024),
    ]

    with patch("psutil.process_iter", return_value=mock_procs):
        summary = audit_active_processes()

    assert summary.browsers_mb == pytest.approx(310.0, rel=1e-2)
    assert summary.ides_mb == pytest.approx(620.0, rel=1e-2)
    assert summary.dev_runtimes_mb == pytest.approx(325.0, rel=1e-2)
    assert summary.system_mb == pytest.approx(200.0, rel=1e-2)
    assert summary.other_mb == pytest.approx(260.0, rel=1e-2)
    assert summary.total_scanned_mb == pytest.approx(1715.0, rel=1e-2)


def test_token_collision_prevention():
    """Verify exact token-delimited matching prevents false positives like codec or mongodb."""
    mock_procs = [
        MockProcess(601, "codec.exe", 50 * 1024 * 1024),          # contains 'code' as substring
        MockProcess(602, "mongodb.exe", 75 * 1024 * 1024),        # contains 'go' as substring
        MockProcess(603, "xcodebuild", 60 * 1024 * 1024),         # contains 'code' as substring
        MockProcess(604, "googleupdate.exe", 40 * 1024 * 1024),   # contains 'google' as substring
        MockProcess(605, "archiver.exe", 30 * 1024 * 1024),       # contains 'arc' as substring
        MockProcess(606, "tor_browser.exe", 80 * 1024 * 1024),    # contains 'tor' as distinct token
    ]

    with patch("psutil.process_iter", return_value=mock_procs):
        summary = audit_active_processes()

    # Only tor_browser should match BROWSERS (token 'tor' in BROWSERS_SET)
    assert summary.browsers_mb == pytest.approx(80.0, rel=1e-2)
    # None of codec, xcodebuild should match IDES
    assert summary.ides_mb == 0.0
    # mongodb should not match DEV_RUNTIMES
    assert summary.dev_runtimes_mb == 0.0
    # codec, mongodb, xcodebuild, googleupdate, archiver all in OTHER
    assert summary.other_mb == pytest.approx(255.0, rel=1e-2)


def test_top_processes_sorted_and_clamped():
    """Verify top_processes extracts strictly top 5 processes by RSS in descending order."""
    mock_procs = [
        MockProcess(1, "proc_100", 100 * 1024 * 1024),
        MockProcess(2, "proc_500", 500 * 1024 * 1024),
        MockProcess(3, "proc_300", 300 * 1024 * 1024),
        MockProcess(4, "proc_700", 700 * 1024 * 1024),
        MockProcess(5, "proc_200", 200 * 1024 * 1024),
        MockProcess(6, "proc_900", 900 * 1024 * 1024),
        MockProcess(7, "proc_400", 400 * 1024 * 1024),
    ]

    with patch("psutil.process_iter", return_value=mock_procs):
        summary = audit_active_processes()

    assert len(summary.top_processes) == 5
    expected_order = ["proc_900", "proc_700", "proc_500", "proc_400", "proc_300"]
    actual_order = [p["name"] for p in summary.top_processes]
    assert actual_order == expected_order

    expected_rss = [900.0, 700.0, 500.0, 400.0, 300.0]
    actual_rss = [p["rss_mb"] for p in summary.top_processes]
    assert actual_rss == expected_rss

    for p in summary.top_processes:
        assert "pid" in p
        assert "category" in p
        assert isinstance(p["category"], str)


def test_defensive_error_handling_and_none_guards():
    """Verify resilience against NoSuchProcess, AccessDenied, ZombieProcess, and None values."""
    mock_procs = [
        MockProcess(10, "valid_chrome.exe", 150 * 1024 * 1024),
        MockProcess(11, "dead_proc", 100 * 1024 * 1024, raise_exc=psutil.NoSuchProcess(11)),
        MockProcess(12, "locked_proc", 100 * 1024 * 1024, raise_exc=psutil.AccessDenied(12)),
        MockProcess(13, "zombie_proc", 100 * 1024 * 1024, raise_exc=psutil.ZombieProcess(13)),
        MockProcess(14, "no_mem_proc", None),
        MockProcess(15, "", 50 * 1024 * 1024),  # Empty name
    ]

    with patch("psutil.process_iter", return_value=mock_procs):
        summary = audit_active_processes()

    # Only valid_chrome (150 MB) and empty name (50 MB) should be counted
    assert summary.browsers_mb == pytest.approx(150.0, rel=1e-2)
    assert summary.other_mb == pytest.approx(50.0, rel=1e-2)
    assert summary.total_scanned_mb == pytest.approx(200.0, rel=1e-2)


def test_live_audit_active_processes():
    """Live integration test: audit_active_processes on real host machine."""
    summary = audit_active_processes()

    assert isinstance(summary, ProcessMemorySummary)
    assert summary.total_scanned_mb > 0.0
    assert len(summary.top_processes) > 0
    assert len(summary.top_processes) <= 5

    # Top processes must be sorted descending by rss_mb
    rss_values = [p["rss_mb"] for p in summary.top_processes]
    assert rss_values == sorted(rss_values, reverse=True)

    for item in summary.top_processes:
        assert isinstance(item["name"], str)
        assert isinstance(item["rss_mb"], float)
        assert item["rss_mb"] >= 0.0
        assert isinstance(item["category"], str)
        assert isinstance(item["pid"], int)

    # All category fields are valid floats
    assert summary.browsers_mb >= 0.0
    assert summary.ides_mb >= 0.0
    assert summary.dev_runtimes_mb >= 0.0
    assert summary.system_mb >= 0.0
    assert summary.other_mb >= 0.0
