"""Catalog provider: a *planning-only* price list of rentable GPU machines.

IMPORTANT: prices below are rough, hand-entered illustrative on-demand USD/hour figures, NOT live quotes
and NOT tied to any specific vendor's current pricing. Use them to compare options and size a budget;
check the real price with your provider. Edit `PRICES` or pass `compute.options.prices` to override.
Real provisioning is not implemented here: `provision` raises. Use `ssh` with a machine you rented.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from bitstraining.compute.base import (
    ComputeProvider,
    Instance,
    JobHandle,
    JobSpec,
    Offer,
    ResourceRequest,
)
from bitstraining.core.errors import ComputeError
from bitstraining.hardware.database import GPU_DB, synthetic_profile

# gpu key -> illustrative USD per GPU-hour (order-of-magnitude priors)
PRICES: dict[str, float] = {
    "t4": 0.35, "l4": 0.70, "a10g": 1.00, "rtx 4090": 0.45, "l40s": 1.50,
    "a100-40gb": 1.80, "a100-80gb": 2.50, "h100": 3.50, "h200": 4.50,
}


class CatalogProvider(ComputeProvider):
    name = "catalog"

    def __init__(self, options: dict[str, Any] | None = None):
        self.prices = {**PRICES, **(options or {}).get("prices", {})}

    def list_offers(self, req: ResourceRequest) -> list[Offer]:
        offers = []
        for gpu, price in sorted(self.prices.items(), key=lambda kv: kv[1]):
            row = GPU_DB.get(gpu)
            if row is None:
                continue
            n = max(req.n_gpus, 1)
            if req.gpu and req.gpu.lower() not in gpu:
                continue
            if row["vram_gb"] < req.min_vram_gb:
                continue
            total = price * n
            if req.max_price_per_hour is not None and total > req.max_price_per_hour:
                continue
            hw = synthetic_profile(gpu, n, total)
            hw.provider = self.name
            offers.append(Offer(provider=self.name, name=f"{n}x {gpu} (illustrative price)", gpu=gpu,
                                n_gpus=n, vram_gb=row["vram_gb"], price_per_hour=round(total, 2), hardware=hw,
                                meta={"illustrative": True}))
        return offers

    def _no(self) -> ComputeError:
        return ComputeError("the catalog provider is planning-only and cannot start machines",
                            fix="rent a machine from your provider and use compute.provider: ssh")

    def provision(self, offer: Offer) -> Instance:
        raise self._no()

    def sync_workspace(self, inst: Instance, bundle_dir: Path) -> None:
        raise self._no()

    def launch(self, inst: Instance, job: JobSpec) -> JobHandle:
        raise self._no()

    def status(self, job: JobHandle) -> str:
        raise self._no()

    def stream_logs(self, job: JobHandle) -> Iterator[str]:
        raise self._no()

    def fetch_artifacts(self, job: JobHandle, dest: Path) -> None:
        raise self._no()

    def cancel(self, job: JobHandle) -> None:
        raise self._no()

    def teardown(self, inst: Instance) -> None:
        pass
