"""Detect the local machine and optionally benchmark it.

Detection uses only the standard library (`nvidia-smi`, `/proc`, `sysctl`). The matmul
microbenchmark needs PyTorch and is skipped (with a note) when torch is absent.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from bitstraining.core.schemas import GPUInfo, HardwareProfile
from bitstraining.hardware.database import lookup_gpu


def _run(cmd: list[str], timeout: float = 10.0) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _cpu_model() -> str:
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text(errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def _ram_from_proc() -> float | None:
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1024 / 1024
    return None


def _ram_from_sysctl() -> float | None:
    out = _run(["sysctl", "-n", "hw.memsize"])  # macOS / BSD
    return int(out) / 1024**3 if out and out.isdigit() else None


def _ram_from_windows() -> float | None:
    if platform.system() != "Windows":
        return None
    try:
        import ctypes

        class _MemStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        st = _MemStatus()
        st.dwLength = ctypes.sizeof(_MemStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):  # type: ignore[attr-defined]
            return st.ullTotalPhys / 1024**3
    except (AttributeError, OSError, ImportError):
        pass
    return None


def _ram_from_sysconf() -> float | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3  # type: ignore[attr-defined]
    except (AttributeError, ValueError, OSError):
        return None


_RAM_PROBES = (_ram_from_proc, _ram_from_sysctl, _ram_from_windows, _ram_from_sysconf)


def _ram_gb() -> float:
    """Total system RAM in GB, or 0.0 if no platform probe works (callers must handle that)."""
    for probe in _RAM_PROBES:
        try:
            v = probe()
        except Exception:  # a broken probe must not stop profiling
            v = None
        if v and v > 0:
            return round(v, 2)
    return 0.0


def _nvidia_gpus() -> list[GPUInfo]:
    if not shutil.which("nvidia-smi"):
        return []
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out:
        return []
    gpus: list[GPUInfo] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        name = parts[0]
        try:
            vram = round(float(parts[1]) / 1024, 2)
        except ValueError:
            continue
        info = lookup_gpu(name, vram_gb=vram)
        if len(parts) > 2 and parts[2]:
            info.compute_capability = parts[2]
            try:
                major = int(parts[2].split(".")[0])
                info.bf16 = info.bf16 or major >= 8
                info.fp8 = info.fp8 or major >= 9
            except ValueError:
                pass
        info.source = "detected" if info.peak_tflops_bf16 == 0 else "database"
        gpus.append(info)
    topo = _run(["nvidia-smi", "topo", "-m"])
    if topo and len(gpus) > 1 and "NV" in topo:
        for g in gpus:
            g.nvlink = True
    return gpus


def _apple_gpu(ram_gb: float) -> list[GPUInfo]:
    if platform.system() != "Darwin" or platform.machine() not in {"arm64", "aarch64"}:
        return []
    brand = _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or "Apple Silicon"
    info = lookup_gpu(brand)
    info.name = brand
    info.vram_gb = round(ram_gb * 0.66, 2)  # default Metal working-set cap is about 2/3 of RAM
    info.source = "database"
    return [info]


def _disk_free_gb(path: str) -> float:
    try:
        return round(shutil.disk_usage(path).free / 1024**3, 2)
    except OSError:
        return 0.0


def _disk_read_gbs(directory: str, size_mb: int = 64) -> float | None:
    """Sequential read speed of the filesystem holding `directory` (page cache may inflate it)."""
    try:
        with tempfile.NamedTemporaryFile(dir=directory, delete=True) as f:
            chunk = os.urandom(1024 * 1024)
            for _ in range(size_mb):
                f.write(chunk)
            f.flush()
            os.fsync(f.fileno())
            f.seek(0)
            t0 = time.perf_counter()
            while f.read(8 * 1024 * 1024):
                pass
            dt = max(time.perf_counter() - t0, 1e-6)
        return round(size_mb / 1024 / dt, 2)
    except OSError:
        return None


def _torch_benchmark(gpus: list[GPUInfo]) -> list[str]:
    """Measure matmul TFLOPs and host->device bandwidth with torch. Returns notes."""
    notes: list[str] = []
    try:
        import torch
    except ImportError:
        return ["benchmark skipped: PyTorch not installed (pip install 'bitstraining[train]')"]
    if not torch.cuda.is_available():
        return ["benchmark skipped: no CUDA device visible to PyTorch"]
    dev = torch.device("cuda:0")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    n = 8192
    a = torch.randn(n, n, device=dev, dtype=dtype)
    b = torch.randn(n, n, device=dev, dtype=dtype)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    iters = 20
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    tflops = 2 * n**3 * iters / dt / 1e12
    host = torch.empty(256 * 1024 * 1024, dtype=torch.uint8).pin_memory()
    t0 = time.perf_counter()
    for _ in range(4):
        host.to(dev, non_blocking=True)
    torch.cuda.synchronize()
    h2d = 4 * host.numel() / (time.perf_counter() - t0) / 1e9
    g = gpus[0]
    # Measured matmul is what the planner calls "peak"; MFU is applied on top.
    g.peak_tflops_bf16 = round(tflops, 1)
    g.h2d_gbs = round(h2d, 1)
    g.source = "measured"
    notes.append(f"measured {tflops:.0f} TFLOPs ({dtype}) and {h2d:.1f} GB/s host->device")
    return notes


def profile(benchmark: bool = False, workdir: str | None = None) -> HardwareProfile:
    """Profile this machine. `benchmark=True` runs the (short) microbenchmarks."""
    ram = _ram_gb()
    gpus = _nvidia_gpus()
    vendor = "nvidia" if gpus else "none"
    unified = False
    if not gpus:
        gpus = _apple_gpu(ram)
        if gpus:
            vendor, unified = "apple", True
    workdir = workdir or os.getcwd()
    hp = HardwareProfile(
        name=platform.node() or "local",
        os=f"{platform.system()} {platform.release()}",
        cpu_model=_cpu_model(),
        cpu_cores=os.cpu_count() or 1,
        ram_gb=ram,
        disk_free_gb=_disk_free_gb(workdir),
        gpu_vendor=vendor,
        unified_memory=unified,
        gpus=gpus,
        provider="local",
    )
    if benchmark:
        hp.disk_read_gbs = _disk_read_gbs(workdir)
        if gpus and vendor == "nvidia":
            _torch_benchmark(gpus)
            hp.measured = any(g.source == "measured" for g in gpus)
    return hp
