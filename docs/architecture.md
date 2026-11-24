# Architecture

## The loop

```
profile ──► plan ──► doctor ──► probe/autotune ──► run ──► observe ──► (calibration feeds the next plan)
```

Everything between the stages is plain, serializable data: `RunSpec` (what you asked for),
`HardwareProfile`, `ModelInfo`, `DatasetStats`, `ExecutionPlan` (what will be done, with its predicted
`Estimate`). A plan can be saved (`plan --save`), diffed, shipped inside a bundle and replayed.

## Packages

| Package | Responsibility | Imports heavy ML libs? |
|---|---|---|
| `core` | schemas, errors with fix suggestions, config loading, plugin registry | no |
| `hardware` | detection (nvidia-smi, /proc, sysctl), GPU capability table, microbenchmark | torch only inside `--benchmark` |
| `models` | exact parameter counts and shapes from a config, never from weights | no |
| `data` | loaders, schema detection, cleaning, dedup, split, packing, stats | no (pyarrow/datasets optional) |
| `planner` | memory model, throughput model, search, doctor, calibration | no |
| `autotune` | micro-batch search, tuning cache, calibration learning, engine bake-off | no |
| `engines` | `Engine` interface + adapters | lazily, inside methods |
| `compute` | `ComputeProvider` interface, providers, bundle, orchestrator | no |
| `tracking` | SQLite registry, lineage, spike tracing | no |
| `eval` | metrics, audit, forgetting gate | generation helper is lazy |
| `optimize`, `deploy` | merge, GGUF, serve commands, model cards | lazily |
| `interfaces` | Python API and CLI | no |

Rule: nothing in core, planner or compute imports torch, transformers or peft. A test enforces that
listing plugins imports none of them.

## The memory model

Per device, peak = weights + master + adapters + gradients + optimizer + activations + logits + context.

* **weights**: parameters at storage precision (bf16 = 2 B, NF4 = 0.53 B including block constants).
* **gradients / optimizer**: per *trainable* parameter. Adapters are kept in fp32 (4 B weight, 4 B grad,
  8 B AdamW state). Full fine-tuning adds fp32 master weights and bf16 gradients.
* **activations**: Korthikanti et al. (2022), `s*b*h*(34 + 5*a*s/h)` bytes per layer, where the
  attention-score term vanishes with fused attention. Split into attention (15) and MLP (19, scaled by
  the intermediate size) parts. Checkpointing keeps only `2*s*b*h` per layer plus one live layer.
* **logits**: `b*s*V*10 B` unfused (bf16 logits, fp32 upcast, fp32 gradient) vs about 1.5 B with a fused
  chunked loss. For a 128k vocabulary this term is large and often forgotten.
* **sharding**: FSDP divides weights, gradients, master and optimizer by the GPU count and adds a
  gathered layer. Layer streaming keeps embeddings, adapters and two layer buffers resident and moves
  the frozen stack to host RAM.

Estimates are multiplied by a learned `mem_ratio` per `(hardware, engine, method)`.

## The throughput model

`time/token = FLOPs/token / (peak * MFU * batch_saturation) * method_overhead`.
FLOPs/token = `multiplier * (2 * matmul_params + attention)` where the multiplier is 3 for full
fine-tuning (forward plus two backward matmuls) and 2 for frozen-base methods, plus recompute for
checkpointing. Padding is modeled from the *measured* length distribution of your data: useful
tokens/s = computed tokens/s x (useful / padded tokens) for the chosen batching mode (padded,
length-grouped, or packed). Layer streaming is bounded by host-to-device bandwidth.
Estimates are multiplied by a learned `speed_ratio`.

## Confidence

* `low`: not calibrated on this hardware, or the GPU is not in the table. Treat numbers as ballpark.
* `medium`: a real probe has calibrated this `(hardware, engine, method)`.
* `high`: calibrated, hardware *measured* with `profile --benchmark`, architecture known exactly, and at
  least 15% memory headroom.

## Orchestration guarantees

`run_job` (compute/orchestrator.py): provision, sync bundle, launch, stream logs, enforce the budget cap
and wall-clock cap, resume after preemption (default 2 times), fetch artifacts, always tear down.
A job that cannot afford five minutes within the budget is refused before anything is provisioned.

## Calibration and the tuning cache

`autotune` probes real steps, compares measured peak memory and tokens/s to the *uncalibrated*
prediction, and folds the ratio into `~/.bitstraining/calibration.json` (exponential moving average,
clamped to 0.4-2.5 so one bad probe cannot poison later plans). Tuning results are cached by hardware,
model, strategy and sequence length, and re-verified with one probe on reuse.
