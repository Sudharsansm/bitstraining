<p align="center">
  <img src="docs/bitstraining.png" alt="Bitstraining" width="160">
</p>

<h1 align="center">Bitstraining</h1>

<p align="center"><b>Train. Fine-tune. Optimize. Deploy.</b></p>

<p align="center">
  <a href="https://pypi.org/project/bitstraining/"><img src="https://img.shields.io/pypi/v/bitstraining" alt="PyPI version"></a>
  <a href="https://pypi.org/project/bitstraining/"><img src="https://img.shields.io/pypi/pyversions/bitstraining" alt="Python versions"></a>
  <a href="LICENSE"><img src="https://img.shields.io/pypi/l/bitstraining" alt="License"></a>
</p>

Bitstraining is a hardware-aware orchestration layer for fine-tuning language models. You describe a
*goal and constraints* (model, data, time or budget); it profiles the hardware, **predicts memory,
speed and cost for many candidate strategies, ranks them**, checks your setup before any GPU time is
spent, verifies the winner with a short real probe, runs it locally or on other compute, and tracks
everything that happened.

It does not reimplement training. PyTorch / Transformers / PEFT, Unsloth and MLX stay swappable
engines behind one small interface. The part that is Bitstraining's own is the planner, the compute
orchestration, the data tooling, evaluation gate, and run tracking.

> **Read [Status & honesty](#status--honesty) before relying on any number.** The planner's
> predictions are model-based estimates that have **not** been validated against real GPU runs yet, and
> nothing here has been benchmarked against other tools. The mechanism is built to make that
> measurement easy (`bitstraining autotune`), not to assume the answer.

```
 goal + constraints ─► hardware profile ─► cost model ─► ranked plans ─► doctor ─► probe/autotune
                                                                                        │
   model card ◄─ eval gate ◄─ tracked run ◄─ engine (HF/PEFT, Unsloth, MLX) ◄─ compute (local/ssh/docker/notebook)
```

## Install

```bash
pip install bitstraining                  # core: planner, data tools, tracking, orchestration, eval. Needs only PyYAML.
pip install "bitstraining[train]"         # + torch, transformers, peft, datasets, accelerate  (real training)
pip install "bitstraining[train,quant]"   # + bitsandbytes (QLoRA, 8-bit optimizers)
pip install "bitstraining[fast]"          # + unsloth  |  "[mlx]" for Apple Silicon  |  "[parquet]"
```

(From a source checkout, use `pip install .` and `pip install ".[train]"` instead. Docker is never
required; the optional `docker` compute provider is only used if you select it.)

Python 3.10+ (the CPU test suite passes on 3.10, 3.11, 3.12 and 3.13 on Linux).

With [uv](https://docs.astral.sh/uv/) (verified: `uv venv`, `uv pip install -e ".[dev]"`, `uv run`, `uv tool install`):

```bash
uv venv && uv pip install -e ".[dev]"
uv run pytest
uv tool install .             # puts `bitstraining` and `bt` on your PATH
```

**Platforms:** developed and tested on Linux. A run on Windows 11 found and fixed a RAM-detection bug (see
CHANGELOG); Windows has not been re-verified since, and macOS has not been tested. Please report problems.

## Try it in 30 seconds, no GPU needed

```bash
bitstraining quickstart
```

This runs the whole pipeline (load data, plan, doctor, run, track) with the **simulation engine**,
which trains nothing and labels every artifact `simulated`. It exists so you can see the system work
and so CI can test orchestration without hardware.

## A real workflow

```bash
bitstraining init --template chat -o run.yaml     # starter config; edit model.base and data.train
bitstraining data inspect data/train.jsonl        # schema, token stats, truncation, padding waste
bitstraining profile                              # what hardware you actually have

bitstraining plan -c run.yaml                                  # ranked plans for THIS machine
bitstraining plan -c run.yaml --gpu a100-80gb --price 2.5      # what-if for rented hardware, with cost
bitstraining doctor -c run.yaml                                # pre-flight: errors come with fixes
bitstraining autotune -c run.yaml                              # real probe: finds the micro-batch, calibrates the model
bitstraining train -c run.yaml --tune                          # plan -> check -> run -> track

bitstraining runs list                            # history
bitstraining runs spikes                          # trace a loss spike back to the training rows
bitstraining eval --task preds.jsonl --general general.jsonl   # per-row audit + forgetting gate
bitstraining card                                 # model card with data hash and lineage
bitstraining serve --model ./output --backend vllm             # prints/starts an OpenAI-compatible server
```

`run.yaml`:

```yaml
name: support-bot
model: { base: meta-llama/Llama-3.1-8B-Instruct }
data:  { train: ./data/train.jsonl, max_length: 1024, eval_split: 0.05 }
goal:  { task: sft, epochs: 3, global_batch_size: 16, objective: time, budget_usd: 20 }
compute: { provider: local }      # or ssh / docker / notebook (see below)
output: ./output
```

## What it does

| Area | Capability |
|---|---|
| **Planner** | Analytic GPU-memory model (weights, adapters, gradients, optimizer, activations, logits, context) and roofline throughput model. Searches method (full / LoRA / QLoRA / DoRA), activation checkpointing, optimizer (AdamW / 8-bit / paged), micro-batch, packing, attention kernel, FSDP/DDP. Ranks by time, cost or memory and respects budgets. |
| **Honest uncertainty** | Every plan carries a confidence level. It is `low` until a real probe has calibrated the model on your hardware. |
| **Doctor** | ~30 pre-flight checks (config sanity, schema/task mismatch, truncation, duplicates, train/eval leakage, resource fit, missing packages) each with a concrete fix. |
| **Autotune** | OOM-safe binary search for the largest micro-batch (a good hint costs 2-3 probes); learns measured/predicted ratios per hardware+engine+method; caches results. |
| **Data** | JSONL/JSON/CSV/TSV/TXT/Parquet/HF; auto schema detection (alpaca, chat, ShareGPT, prompt/completion, preference, raw text); cleaning; exact + MinHash near-duplicate removal; leakage-safe splits; packing and padding-waste analysis. |
| **Compute** | One job description, many places: `local` (subprocess), `ssh` (any GPU box), `docker`, `notebook` (self-running Colab/Kaggle notebook), `catalog` (illustrative price list for planning). Hard budget cap, wall-clock cap, auto-resume after preemption, guaranteed teardown. |
| **Tracking** | SQLite registry: runs, plans, metrics, artifacts with hashes, lineage from data hash to model. Per-step batch row ids let a loss spike be traced to the rows behind it. |
| **Eval** | Metrics, per-row better/worse/same audit, forgetting check, and a gate that blocks export on regressions. Pluggable judge: any `(prediction, reference) -> float`. |
| **Deploy** | Model cards, merge LoRA into base, GGUF export via llama.cpp, serve via vLLM or llama.cpp. |

## Compute targets

```yaml
compute: { provider: ssh, options: { host: ubuntu@203.0.113.7, gpu: a100-80gb, price_per_hour: 1.9 } }
compute: { provider: docker, options: { image: pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime } }
compute: { provider: notebook, options: { target: colab } }   # writes a notebook; you press Run all
```

The runner ships inside a *bundle* (spec, plan, data, a copy of this package), so remote machines
don't need Bitstraining installed. Colab and Kaggle cannot be driven unattended, so the notebook provider is
deliberately manual: it embeds the bundle in one notebook and you download `output.zip` afterwards
(`bitstraining compute fetch --zip output.zip`).

## Extending

Engines and providers are plugins, discovered through Python entry points (or `registry.register(...)`
at runtime). See [docs/plugins.md](docs/plugins.md) and `examples/custom_provider.py`.

## Status & honesty

This table is the source of truth. "Verified" means exercised by the automated test suite
(the CPU-only test suite, including real subprocess jobs); it does **not** mean validated on GPUs.

| Component | State |
|---|---|
| Planner (memory, throughput, search, doctor), calibration, autotune loop | **Verified** logic; parameter counts checked against published sizes (<1% error for 7 models). **Not validated** against real GPU measurements. |
| Data pipeline, tracking, eval/gate, model cards, orchestrator, bundle, local subprocess provider, notebook builder | **Verified** by tests, including mutation checks on the critical logic. |
| `hf_peft` engine | **Written, never run** against real torch/GPU in this repo (no GPU available during development). Prompt masking and tokenization logic is unit-tested with a stub tokenizer. Treat first use as a smoke test. |
| `unsloth`, `mlx` engines | **Written, never run.** Experimental. |
| `ssh`, `docker` providers | Command construction **unit-tested**; **never run** against a real host/daemon. |
| Layer streaming, FSDP/DeepSpeed, DPO/ORPO/KTO/GRPO training, pruning, distillation | **Not implemented.** The planner can *model* streaming and FSDP; no bundled real engine executes them. Preference tasks are rejected cleanly at planning time. |
| Web UI, MCP server, real cloud provisioning (RunPod, Lambda, AWS...) | **Not implemented.** `catalog` is a planning-only price list with illustrative numbers. |

**On "faster and more efficient".** The efficiency mechanisms that exist are: choosing QLoRA/LoRA vs full
fine-tuning by what fits, activation checkpointing only when memory demands it, 8-bit optimizers,
packing/length-grouping to cut padding waste, largest-safe micro-batch search, and cost-aware hardware
comparison. The planner quantifies their *predicted* effect. **No benchmark here shows Bitstraining
trains faster than Axolotl, LLaMA-Factory, Unsloth or plain Transformers**, and none should be assumed.
Speed-ups quoted by other libraries are theirs, not ours; use `bitstraining autotune` to measure on
your hardware.

Model constants that are heuristics, not measurements (all in one place, all tunable by calibration):
base MFU 0.42, QLoRA time overhead 1.20, DoRA 1.25, layer-streaming overlap loss 0.30, token
estimate of 3.8 characters per token when no tokenizer is available, and the GPU table's peak-FLOP
figures.

## Documentation

* [Architecture](docs/architecture.md): how the planner, engines, compute providers and tracking fit together
* [Writing plugins](docs/plugins.md): custom engines and compute providers (`examples/custom_provider.py`)
* [Releasing](RELEASING.md): publishing to PyPI from GitHub
* [Changelog](CHANGELOG.md): includes which earlier *results* a fix could have affected

## Development

```bash
pip install -e ".[dev]"
pytest                      # a few seconds, no GPU
ruff check bitstraining tests
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

Apache-2.0. Bitstraining was inspired by the single-config, pre-flight-check, measured-numbers ethos of
projects such as Soup CLI (trysoup.dev); it shares no code with them.
