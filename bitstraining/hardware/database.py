"""GPU capability table used when a device cannot be benchmarked (or is remote).

Numbers are public *dense* bf16/fp16 tensor-core peaks and memory bandwidths, rounded.
They are priors: the planner treats database-sourced hardware as lower confidence than
hardware that was measured by `bitstraining profile --benchmark`.
Add or correct entries freely; names are matched case-insensitively by substring.
"""

from __future__ import annotations

from bitstraining.core.schemas import GPUInfo, HardwareProfile

# name -> (vram_gb, peak_tflops_bf16, mem_bw_gbs, compute_capability, bf16, fp8, nvlink, h2d_gbs)
_ROWS: dict[str, tuple[float, float, float, str, bool, bool, bool, float]] = {
    "rtx 3050 laptop 4gb": (4, 18, 192, "8.6", True, False, False, 12),
    "rtx 3050": (8, 36, 224, "8.6", True, False, False, 12),
    "rtx 3060": (12, 51, 360, "8.6", True, False, False, 12),
    "rtx 3070": (8, 81, 448, "8.6", True, False, False, 12),
    "rtx 3080": (10, 119, 760, "8.6", True, False, False, 12),
    "rtx 3090": (24, 142, 936, "8.6", True, False, False, 12),
    "rtx 4060": (8, 60, 272, "8.9", True, True, False, 12),
    "rtx 4070": (12, 117, 504, "8.9", True, True, False, 12),
    "rtx 4080": (16, 195, 717, "8.9", True, True, False, 12),
    "rtx 4090": (24, 330, 1008, "8.9", True, True, False, 12),
    "rtx 5090": (32, 420, 1790, "12.0", True, True, False, 24),
    "t4": (16, 65, 320, "7.5", False, False, False, 12),
    "v100": (32, 112, 900, "7.0", False, False, True, 12),
    "l4": (24, 121, 300, "8.9", True, True, False, 12),
    "a10g": (24, 125, 600, "8.6", True, False, False, 12),
    "a10": (24, 125, 600, "8.6", True, False, False, 12),
    "l40s": (48, 362, 864, "8.9", True, True, False, 24),
    "a100-40gb": (40, 312, 1555, "8.0", True, False, True, 24),
    "a100-80gb": (80, 312, 2039, "8.0", True, False, True, 24),
    "a100": (40, 312, 1555, "8.0", True, False, True, 24),
    "h100": (80, 990, 3350, "9.0", True, True, True, 48),
    "h200": (141, 990, 4800, "9.0", True, True, True, 48),
    "b200": (180, 2250, 8000, "10.0", True, True, True, 48),
    "mi300x": (192, 1307, 5300, "gfx942", True, True, True, 48),
    # Apple Silicon (unified memory; vram_gb is a usable share of system memory)
    "apple m1": (10, 2.6, 68, "mps", False, False, False, 60),
    "apple m2": (12, 3.6, 100, "mps", False, False, False, 80),
    "apple m3": (12, 4.1, 100, "mps", False, False, False, 80),
    "apple m4": (12, 4.6, 120, "mps", False, False, False, 90),
}

GPU_DB = {
    k: dict(
        vram_gb=v[0],
        peak_tflops_bf16=v[1],
        mem_bw_gbs=v[2],
        compute_capability=v[3],
        bf16=v[4],
        fp8=v[5],
        nvlink=v[6],
        h2d_gbs=v[7],
    )
    for k, v in _ROWS.items()
}


def lookup_gpu(name: str, vram_gb: float | None = None) -> GPUInfo:
    """Best-effort match of a device name to the database.

    Longest key contained in `name` wins ("rtx 3050 laptop 4gb" beats "rtx 3050").
    Unknown devices get conservative numbers and `source="database"` with zero peak
    so the planner reports low confidence.
    """
    low = name.lower().replace("nvidia", "").replace("geforce", "").strip()
    low = " ".join(low.split())
    best = None
    for key in sorted(GPU_DB, key=len, reverse=True):
        if key in low or key.replace("-", " ") in low.replace("-", " "):
            best = key
            break
    if best is not None and vram_gb is None:
        # Names like "A100-SXM4-80GB" carry the memory size: prefer the family variant that matches it.
        import re

        m = re.search(r"(\d+)\s*gb", low)
        if m:
            family = best.split("-")[0].split(" ")[0]
            for key, row in GPU_DB.items():
                if key.split("-")[0].split(" ")[0] == family and row["vram_gb"] == float(m.group(1)):
                    best = key
                    break
    if best is None and vram_gb is not None:
        # Pick by closest VRAM as a very rough prior.
        return GPUInfo(
            name=name,
            vram_gb=vram_gb,
            peak_tflops_bf16=0.0,
            mem_bw_gbs=0.0,
            bf16=False,
            source="database",
        )
    if best is None:
        return GPUInfo(name=name, source="database")
    row = dict(GPU_DB[best])
    if vram_gb:
        row["vram_gb"] = vram_gb  # trust the detected size over the table
    return GPUInfo(name=name, source="database", **row)  # type: ignore[arg-type]


def synthetic_profile(
    gpu: str, n_gpus: int = 1, price_per_hour: float | None = None
) -> HardwareProfile:
    """A hardware profile for machines we cannot touch (rented GPUs, what-if planning)."""
    g = lookup_gpu(gpu)
    if g.peak_tflops_bf16 == 0:
        raise ValueError(f"unknown GPU '{gpu}'. Known: {', '.join(sorted(GPU_DB))}")
    g.name = gpu
    return HardwareProfile(
        name=f"{gpu} x{n_gpus} (planned)",
        os="linux",
        cpu_model="planned",
        cpu_cores=max(8, 8 * n_gpus),
        ram_gb=max(64.0, 64.0 * n_gpus),
        disk_free_gb=500.0,
        disk_read_gbs=2.0,
        gpu_vendor="nvidia",
        gpus=[GPUInfo(**g.to_dict()) for _ in range(n_gpus)],
        price_per_hour=price_per_hour,
        provider="planned",
        measured=False,
    )
