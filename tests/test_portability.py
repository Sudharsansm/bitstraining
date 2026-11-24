"""Platform portability: nothing may assume Linux. (Regression for a Windows failure where undetected
RAM became a 0 GB device and every plan was 'infeasible'.)"""

import os
import subprocess
import sys

import pytest

import bitstraining.hardware.profiler as prof
from bitstraining.compute.base import Instance, JobHandle
from bitstraining.compute.local import _PROCS, LocalProvider
from bitstraining.core.schemas import HardwareProfile
from bitstraining.interfaces.cli import main
from bitstraining.planner.doctor import run_doctor
from bitstraining.planner.throughput import UNKNOWN_RAM_GB, device_for


def test_ram_probe_chain_uses_first_working_probe_and_survives_broken_ones(monkeypatch):
    def boom():
        raise OSError("platform API exploded")

    monkeypatch.setattr(prof, "_RAM_PROBES", (lambda: None, boom, lambda: 0.0, lambda: 15.9, lambda: 99.0))
    assert prof._ram_gb() == 15.9
    monkeypatch.setattr(prof, "_RAM_PROBES", (lambda: None, boom))
    assert prof._ram_gb() == 0.0  # "unknown" is reported honestly, never invented


def test_this_platform_reports_nonzero_ram():
    assert prof._ram_gb() > 0


def test_windows_probe_is_inert_elsewhere_and_sysconf_works_on_posix():
    if sys.platform != "win32":
        assert prof._ram_from_windows() is None
    if hasattr(os, "sysconf"):
        assert (prof._ram_from_sysconf() or 0) > 0


def test_undetected_ram_never_makes_a_zero_memory_device():
    hw = HardwareProfile(name="x", ram_gb=0.0, cpu_cores=4)
    assert device_for(hw).vram_gb == pytest.approx(UNKNOWN_RAM_GB * 0.6)


def test_doctor_warns_when_ram_is_unknown(spec):
    rep = run_doctor(spec, HardwareProfile(name="x", ram_gb=0.0), None, None, None, None, check_imports=False)
    assert any(f.code == "DR050" for f in rep.findings)


def test_quickstart_works_even_when_ram_cannot_be_detected(monkeypatch, capsys):
    # The exact failure from the Windows report: no GPU + RAM probe returning 0.
    monkeypatch.setattr(prof, "_ram_gb", lambda: 0.0)
    assert main(["quickstart"]) == 0


def test_local_cancel_works_without_killpg(monkeypatch):
    """Windows has no os.killpg; cancel must fall back to terminate()."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        _PROCS["win-sim"] = proc
        if hasattr(os, "killpg"):
            monkeypatch.delattr(os, "killpg")
        LocalProvider().cancel(JobHandle("win-sim", Instance("i", "local")))
        assert proc.wait(timeout=10) is not None
    finally:
        if proc.poll() is None:
            proc.kill()
        _PROCS.pop("win-sim", None)
