"""Plugin registry. Engines and compute providers are plugins.

Resolution order for a name in a group:
1. objects registered at runtime with `register()`
2. entry points declared by installed packages (group `bitstraining.<group>`)
3. the built-in table below (so a source checkout works without `pip install`)

Targets are stored as "module:Attr" strings and imported lazily, so listing plugins
never imports torch, transformers or any other heavy dependency.
"""

from __future__ import annotations

import importlib
from importlib import metadata
from typing import Any

from bitstraining.core.errors import BitstrainingError

BUILTIN: dict[str, dict[str, str]] = {
    "engines": {
        "simulate": "bitstraining.engines.simulate:SimulateEngine",
        "hf_peft": "bitstraining.engines.hf_peft:HFPeftEngine",
        "unsloth": "bitstraining.engines.unsloth:UnslothEngine",
        "mlx": "bitstraining.engines.mlx:MLXEngine",
        "deepspeed": "bitstraining.engines.deepspeed:DeepSpeedEngine",
    },
    "providers": {
        "local": "bitstraining.compute.local:LocalProvider",
        "ssh": "bitstraining.compute.ssh:SSHProvider",
        "docker": "bitstraining.compute.docker:DockerProvider",
        "notebook": "bitstraining.compute.notebook:NotebookProvider",
        "catalog": "bitstraining.compute.catalog:CatalogProvider",
    },
}

_runtime: dict[str, dict[str, Any]] = {}


def register(group: str, name: str, target: Any) -> None:
    """Register a class/object (or a "module:Attr" string) under group/name."""
    _runtime.setdefault(group, {})[name] = target


def _entry_points(group: str) -> dict[str, str]:
    found: dict[str, str] = {}
    try:
        eps = metadata.entry_points(group=f"bitstraining.{group}")
    except Exception:  # pragma: no cover - metadata backends differ
        return found
    for ep in eps:
        found[ep.name] = ep.value
    return found


def names(group: str) -> list[str]:
    out = dict.fromkeys(BUILTIN.get(group, {}))
    out.update(dict.fromkeys(_entry_points(group)))
    out.update(dict.fromkeys(_runtime.get(group, {})))
    return sorted(out)


def _resolve(target: Any) -> Any:
    if isinstance(target, str):
        module, _, attr = target.partition(":")
        return getattr(importlib.import_module(module), attr)
    return target


def get(group: str, name: str) -> Any:
    """Return the plugin class/object. Raises with the list of known names."""
    if name in _runtime.get(group, {}):
        return _resolve(_runtime[group][name])
    eps = _entry_points(group)
    if name in eps:
        return _resolve(eps[name])
    if name in BUILTIN.get(group, {}):
        return _resolve(BUILTIN[group][name])
    raise BitstrainingError(
        f"no {group[:-1]} named '{name}'",
        fix=f"available: {', '.join(names(group)) or '(none)'}",
    )


def create(group: str, name: str, *args: Any, **kwargs: Any) -> Any:
    """Instantiate a plugin."""
    return get(group, name)(*args, **kwargs)
