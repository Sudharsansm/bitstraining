# Writing plugins

Engines and compute providers are discovered through Python entry points, or registered at runtime.

## A compute provider

Subclass `bitstraining.compute.base.ComputeProvider` and implement ten methods
(`list_offers`, `provision`, `sync_workspace`, `launch`, `status`, `stream_logs`, `fetch_artifacts`,
`cancel`, `teardown`). A job is a *bundle directory*; the provider moves it, starts
`python -m bitstraining.runner --bundle <dir>` (or runs `bootstrap.sh`), and returns the `output/` folder.
`status` must return one of `running`, `succeeded`, `failed`, `preempted` (host lost; the orchestrator
will resume) or `manual` (a human must act; the orchestrator prints instructions and stops).

Register it either way:

```toml
# your pyproject.toml
[project.entry-points."bitstraining.providers"]
myprovider = "my_pkg.provider:MyProvider"
```

```python
from bitstraining.core import registry
registry.register("providers", "myprovider", MyProvider)
```

Then `compute: {provider: myprovider, options: {...}}`. The constructor receives `options`.
See `examples/custom_provider.py` for a complete, tested example.

## An engine

Subclass `bitstraining.engines.base.Engine`:

* `capabilities(hw, assume_installed)`: what you can do on this hardware (methods, precisions,
  optimizers, attention kernels, parallelism, tasks, platforms). The planner searches only inside this.
  Report `available=False` with a `reason` when your library is missing, and give an honest
  `speed_prior` (use 1.0 unless you have measurements; `speed_prior_is_measured` says whether you do).
* `probe(plan, ctx, micro_batch)`: run a few real steps, return peak memory and tokens/s. This is what
  makes autotune and calibration work.
* `train(plan, spec, train_rows, eval_rows, callbacks, output_dir, resume_from)`: rows arrive already
  cleaned and normalized (`{"messages": [...]}`, `{"text": ...}` or preference triples). Call
  `callbacks.step(step, metrics)` every optimizer step and `callbacks.on_batch(step, row_ids)` if you can,
  so loss spikes are traceable to data. Honor `callbacks.stop_requested()`.

Import heavy libraries inside methods, never at module top level.
