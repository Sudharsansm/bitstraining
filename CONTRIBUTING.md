# Contributing to Bitstraining

Thanks for helping. The project's rule is simple: **numbers are measured, not claimed.**

## Setup

```bash
git clone <your fork> && cd bitstraining
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest && ruff check bitstraining tests
```

The suite needs no GPU and runs in seconds. If you add code that needs torch, import it inside the
function, never at module top level (a test enforces that listing plugins imports no heavy library).

## What makes a good change

* **Fix with a test.** Reproduce the bug in a test first. Several real bugs in this codebase were found
  that way (see CHANGELOG).
* **Test the property, not the call.** "Packing returns a valid partition within capacity" beats
  "packing returns a list". Before sending a PR touching critical logic (memory model, orchestrator budget
  cap, eval gate, packing, dedup, spike detection), try breaking it on purpose; if the suite stays green,
  the tests are too weak.
* **Heuristics are labeled.** A constant that is an engineering guess (MFU, overheads, overlap loss)
  lives in one named place with a comment saying so, and calibration can override it.
* **Errors carry fixes.** Raise `BitstrainingError` subclasses with `fix=` text that says what to do.
* **Honest status.** If you add something you could not run (no GPU, no remote host), say so in the module
  docstring and in the README status table. "Written, never run" is acceptable; implying otherwise is not.

## Adding an engine or provider

See [docs/plugins.md](docs/plugins.md). Include unit tests for everything that does not need the real
backend (command construction, data encoding, status mapping) and mark the rest as unverified.

## Performance and accuracy claims

Do not add speed-up claims or comparisons with other tools without a reproducible measurement:
hardware, library versions, command, raw output. Predicted-vs-measured results from `bitstraining
autotune` are welcome as issues; they are how the cost model gets better.

## Pull requests

Small and focused. Describe what changed, how you verified it, and what you could *not* verify.
By contributing you agree your work is licensed under Apache-2.0.
