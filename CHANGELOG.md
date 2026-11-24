# Changelog

Format follows Keep a Changelog; versions follow SemVer. Entries say which earlier *results* could be
affected, because for a planner that matters as much as API changes.

## 0.1.0 - initial release

First working version: planner, data pipeline, autotune, tracking, compute orchestration, eval gate,
CLI and Python API. See the README "Status & honesty" table for what is verified and what is not.

### Bugs found and fixed by the test suite before release
* **Duplicate detection missed whitespace-only differences** at message boundaries (`"Q1 "` vs `"q1"`),
  letting near-identical rows into training and across the train/eval split. Normalization is now per
  message.
* **GPU lookup returned the 40 GB A100 for `A100-SXM4-80GB`** (the string `nvidia-smi` reports) because only
  the exact key `a100-80gb` matched. The memory size in the name is now honored.
* **Layer streaming was modeled as free** when compute exceeded transfer time; an imperfect-overlap term
  (a labeled heuristic) is now included.
* **Plan confidence was overstated**: uncalibrated estimates were reported as "medium". They are now "low"
  until a real probe calibrates the hardware.

### Found by a user run on Windows 11, fixed in 0.1.0
* **RAM was not detected on Windows**, so a machine without a GPU was modeled as a 0 GB device and every
  plan was reported infeasible (6 of 150 tests failed on that machine; the other 144 passed). RAM is now
  read through Windows, POSIX `sysconf`, `/proc` and `sysctl` probes. If none work the planner assumes 8 GB
  and the doctor warns (DR050) instead of silently planning for nothing.
* `LocalProvider.cancel` used `os.killpg` (POSIX only) and would have failed on Windows once planning
  succeeded; it now falls back to `terminate()`.
  Verified by reproducing the failure on Linux (forcing RAM detection to 0) and adding regression tests.
  **Not yet re-run on Windows itself.**

### Known gaps
No validation against real GPU runs; `hf_peft`, `unsloth`, `mlx`, `ssh` and `docker` paths have never
been executed end to end; no layer-streaming or sharded (FSDP/ZeRO) engine; no preference/RL training;
no web UI or MCP server; no real cloud provisioning.
