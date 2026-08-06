# homebench

**Benchmark the local LLMs you already have — speed, memory, *and* quality — as a live terminal leaderboard.**

[![CI](https://github.com/david-g-3654/homebench/actions/workflows/ci.yml/badge.svg)](https://github.com/david-g-3654/homebench/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/homebench)](https://pypi.org/project/homebench/)
![Python](https://img.shields.io/pypi/pyversions/homebench)
![License](https://img.shields.io/pypi/l/homebench)

![homebench demo](https://cdn.jsdelivr.net/gh/david-g-3654/homebench@main/docs/demo.svg)

`homebench` is a single-command TUI that discovers the models installed in your local runner (**Ollama**, **LM Studio**, **llama.cpp**, **vLLM**, or any **OpenAI-compatible** server), runs a curated quality suite, measures **tokens/sec**, **time-to-first-token**, and **memory footprint** on *your actual machine*, and renders a live comparison leaderboard.

```bash
pip install homebench
homebench
```

That's it. No config, no API keys, no cloud.

---

## Why

There are great tools for *one* half of this problem, but nothing local-first that does both:

- [`llama-bench`](https://github.com/ggml-org/llama.cpp) (inside llama.cpp) measures **speed only**.
- [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness) measures **quality** but has no polished laptop UX and isn't built around the model runners most people actually use locally.

`homebench` fills the gap: **local-first, zero-config, UX-driven.** Clone-and-run, point it at the models you already pulled, and get an at-a-glance answer to *"which of my local models is actually good, and how fast is it on this laptop?"*

## What it measures

| Metric | How |
| --- | --- |
| **tok/s** | Output tokens ÷ generation time. Ollama reports server-side eval timing; OpenAI-compatible backends are timed client-side from the token stream. Excludes prompt processing and model load. |
| **TTFT** | Wall-clock time to the first streamed token (minus model-load time where the runner reports it). |
| **Memory** | Resident model size when the runner exposes it (Ollama `/api/ps`, LM Studio `/api/v0`), plus a best-effort peak-RSS sample of the backend's processes. |
| **Quality** | 31 deterministically-graded tasks across math, reasoning, factual recall, instruction-following/structured-output, extraction, and code understanding. Optional **LLM-as-judge** adds open-ended tasks (summaries, email, haiku, explanations). |

## Install

```bash
pip install homebench        # then run:  homebench
```

Prefer an isolated install? Use [pipx](https://pipx.pypa.io):

```bash
pipx install homebench
```

Or from source:

```bash
git clone https://github.com/david-g-3654/homebench
cd homebench
pip install .
```

Requires **Python 3.9+**.

## Usage

```bash
homebench                        # fast default: 3 smallest models, quick suite (TUI)
homebench --all                  # benchmark every discovered model
homebench --full                 # run the full quality suite (not just the fast subset)
homebench --no-tui               # plain live renderer (great for piping / CI)
homebench -m llama3.2,qwen3:8b   # only these models
homebench --limit 3              # cap the number of models
homebench --provider lmstudio    # use LM Studio instead of auto-detect
homebench --provider llamacpp    # llama.cpp server (llama-server)
homebench --provider vllm        # vLLM
homebench --provider openai --host http://localhost:5000   # any OpenAI-compatible server
homebench --refresh-cache        # recompute instead of reusing cached responses
homebench --no-quality           # speed + memory only (fast)
homebench --no-speed             # quality only
homebench --judge qwen3:8b       # enable LLM-as-judge (adds open-ended tasks)
homebench --tasks mypack.yaml    # use a custom task pack instead of the built-in suite
homebench --add-tasks mypack.yaml  # add a pack on top of the built-in suite
homebench --label "before tuning"  # tag this run for later diffing
homebench --md results.md        # also export a Markdown report
homebench --json results.json    # also export raw JSON
homebench --html report.html     # self-contained, shareable HTML report

homebench list                   # just list discovered models
homebench tasks                  # show the quality suite (add --tasks to preview a pack)
homebench history                # list past runs (saved automatically)
homebench diff                   # diff the two most recent runs
homebench diff 3 1               # diff run #3 (base) against run #1 (newer)
homebench report latest --html run.html   # render any saved run as a report
homebench throughput             # batch-throughput sweep (concurrency 1,2,4,8)
homebench throughput --concurrency 1,8,16 --provider vllm
homebench fit                    # which popular models fit YOUR hardware?
```

Run `homebench --help` for the full flag list.

### Example output

A real quick-suite run on an Apple M1 (16 GB), via Ollama:

```
                               Final leaderboard
┏━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃ # ┃ Model                ┃ Params ┃ Quality ┃ Pass ┃ tok/s ┃   TTFT ┃ Memory ┃
┡━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│ 1 │ llama3.2:latest      │   3.2B │     75% │  6/8 │  16.8 │ 545 ms │ 2.4 GB │
│ 2 │ alibayram/smollm3    │   3.1B │     38% │  3/8 │  16.9 │ 829 ms │ 2.1 GB │
└───┴──────────────────────┴────────┴─────────┴──────┴───────┴────────┴────────┘
```

(Numbers are for *that* laptop at *that* moment — see [Limitations](#limitations).)

## Providers

At least one local model runner must be reachable:

| Provider | `--provider` | Default host | Host env var | Notes |
| --- | --- | --- | --- | --- |
| Ollama | `ollama` | `http://localhost:11434` | `OLLAMA_HOST` | Native API; reports model memory via `/api/ps`. |
| LM Studio | `lmstudio` | `http://localhost:1234` | `LMSTUDIO_HOST` | Enriches metadata + memory via native `/api/v0`. |
| llama.cpp | `llamacpp` | `http://localhost:8080` | `LLAMACPP_HOST` | `llama-server`, OpenAI-compatible. |
| vLLM | `vllm` | `http://localhost:8000` | `VLLM_HOST` | Set `VLLM_API_KEY` if started with `--api-key`. |
| OpenAI-compatible | `openai` | — | `OPENAI_BASE_URL` | Any `/v1` server (Jan, LocalAI, TGI, …); pass `--host`. |

Auto-detection tries Ollama → LM Studio → llama.cpp → vLLM (the generic `openai` provider is explicit-only). Force one with `--provider`. Override host with `--host` or the env var above.

## How quality grading works

The suite is small on purpose — enough tasks across categories to *separate* models, few enough that every model runs in a couple of minutes on a laptop. Each task is graded deterministically (exact numeric match, multiple-choice letter, substring, valid-JSON, regex). Temperature is 0 and a fixed seed is used for reproducibility. See `homebench tasks` for the list.

The optional `--judge MODEL` flag turns on an LLM-as-judge (any local model) that scores open-ended tasks 1–5 against a reference answer. It's a signal, not an oracle.

### Fast by default

Benchmarking every model on the full suite takes a while on a laptop, so the defaults are tuned for a quick first look:

- **3 smallest models** by default (smallest first, so results appear fast) — `--all` for everything, `-m` to choose.
- **A fast quality subset** (~8 tasks across all categories) — `--full` for all 31.
- **Response caching**: quality runs use temperature 0 + a fixed seed, so responses are deterministic and cached under `~/.homebench`. Re-running only regenerates *new* models/tasks (unchanged ones are re-graded from cache in milliseconds); `--refresh-cache` forces recompute, `--no-cache` disables it.

In practice this turns a first run from ~15–25 min (all models, full suite) into ~1–2 min, and a re-run into seconds. For a thorough pass (CI, final numbers) use `homebench --all --full`.

## Custom task packs

Bring your own evals with a JSON or YAML pack — no Python required. `--tasks` replaces the built-in suite; `--add-tasks` appends to it. YAML needs the optional extra (`pip install "homebench[yaml]"`); JSON works out of the box.

```yaml
# mypack.yaml  —  homebench --tasks mypack.yaml
name: my-pack
tasks:
  - id: capital_japan
    category: factual
    prompt: "What is the capital of Japan? Answer with just the city name."
    grader: {type: contains_any, values: ["Tokyo"]}
    reference: Tokyo
  - id: add
    category: math
    prompt: "What is 12 + 30? End with the answer on its own line."
    grader: {type: exact_number, value: 42}
  - id: explain          # no grader -> open-ended, scored only with --judge
    category: open
    prompt: "Explain photosynthesis in one sentence."
    reference: "Plants convert sunlight, water, and CO2 into glucose and oxygen."
```

Grader `type` values: `exact_number` (`value`, `tol`), `multiple_choice` (`value`), `contains_any` (`values`), `regex` (`pattern`, `ignorecase`), `valid_json` (`keys`), `valid_json_array` (`length`). Omit `grader` for a judge-only task. Runnable examples live in [`examples/`](examples/); preview any pack with `homebench tasks --tasks mypack.yaml`.

## History & diffing

Every run is saved automatically to `$HOMEBENCH_HOME/runs` (default `~/.homebench/runs`); disable with `--no-save`, and tag runs with `--label`.

```bash
homebench history            # table of past runs (newest first)
homebench diff               # previous run -> latest
homebench diff 3             # run #3 -> latest
homebench diff 3 1           # run #3 (base) -> run #1 (newer)
```

`diff` compares models by name and shows per-model deltas in quality and throughput, plus which models were added or removed between runs — handy for "did that quantization / setting actually help?" Every run also captures the **environment** (OS, CPU, RAM, GPU, homebench + Python versions), so reports are reproducible and `diff` warns when two runs came from different machines.

### Shareable reports

Export a run as a **self-contained HTML page** (inline styles, CSS bars, theme-aware — no external assets, safe to email or drop in a gist):

```bash
homebench --html report.html          # from a fresh run
homebench report latest --html run.html   # or render any saved run (also --md)
```

## Batch throughput

The main leaderboard measures **single-stream** tok/s. Servers that batch requests (vLLM, llama.cpp continuous batching, Ollama with `OLLAMA_NUM_PARALLEL>1`) can do far more total work under concurrency — `homebench throughput` measures that:

```bash
homebench throughput -m my-model --concurrency 1,2,4,8
```

It fires N requests at each concurrency level (N defaults to 3×concurrency) and reports **aggregate** tok/s (total output ÷ wall-clock), the speedup vs. concurrency 1, mean per-request rate, and latency (mean / p95):

```
             Batch throughput — my-model (vllm)
┏━━━━━━┳━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┓
┃ Conc ┃ Reqs ┃ Agg tok/s ┃ Speedup ┃ Req tok/s ┃ Mean lat ┃ p95 lat ┃ Errors ┃
┡━━━━━━╇━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━┩
│    1 │    4 │      95.0 │   1.00× │      95.0 │   1.35 s │  1.4 s  │      0 │
│    4 │   12 │     320.0 │   3.37× │      82.0 │   1.56 s │  1.9 s  │      0 │
│    8 │   24 │     540.0 │   5.68× │      70.0 │   1.83 s │  2.6 s  │      0 │
└──────┴──────┴───────────┴─────────┴───────────┴──────────┴─────────┴────────┘
```

On a non-batching setup, aggregate throughput stays flat while latency climbs — which is itself a useful thing to see. Add `--json FILE` to export.

## What can my machine run?

![homebench fit](https://cdn.jsdelivr.net/gh/david-g-3654/homebench@main/docs/fit-demo.svg)

Before benchmarking, `homebench fit` captures your hardware (RAM, CPU, GPU/VRAM, Apple unified memory) and checks a **catalog of ~50 popular models** — SmolLM2, Qwen2.5, Llama 3.x, Gemma 2, Phi-3.5/4, Mistral/Mixtral, DeepSeek-R1, CodeLlama, Yi, Command-R, and more, from 135M up to 141B — against your memory budget, showing which fit and at what quantization:

```bash
homebench fit                    # what fits, at the best quant
homebench fit --all              # include models that don't fit
homebench fit --context 8192     # budget a larger KV cache
homebench fit --quant Q4_K_M     # evaluate a specific quant
homebench fit --vram 24          # what-if: "if I had a 24 GB GPU…"
homebench fit --catalog my.json  # add your own models to the catalog
```

### Live list from HuggingFace

Instead of the built-in catalog, pull the **currently most popular models straight from the HuggingFace Hub** — their parameter counts (from safetensors metadata) are sized against your hardware in real time:

```bash
homebench fit --online              # top 50 text-generation models by downloads
homebench fit --online --top 100    # cast a wider net
homebench fit --online --sort trending   # or: likes
homebench fit --online --refresh    # bypass the 1-day cache
```

Results are cached under `$HOMEBENCH_HOME` (`~/.homebench`), so repeat runs are fast and work offline; if the Hub is unreachable, `homebench` falls back to the cache (or the built-in catalog).

The built-in catalog also ships each model's **Ollama tag** (`ollama pull …`) and **HuggingFace repo** (which LM Studio and vLLM pull from). Add your own with a JSON catalog (see [`examples/models.example.json`](examples/models.example.json)): a list of `{name, params_b, family?, ollama?, hf?}`. Sizes are estimates (weights + KV cache + overhead), so treat "fits"/"tight" as guidance. Add `--json FILE` to export the hardware profile and results.

## Limitations

`homebench` is a fast, local **first look** — not a rigorous benchmark of record. Keep these in mind:

- **Quality is a signal, not a leaderboard of record.** The suite is small and English-only (8 tasks in the fast default, 31 with `--full`); it's designed to *separate* your models, not to rank them authoritatively. For serious evals use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). The optional LLM-as-judge is noisy, especially with small local judges.
- **Speed is your-machine-at-that-moment.** tok/s and TTFT depend on current load, thermal state, and memory pressure — a busy laptop (or swapping when low on RAM) will read slower. Numbers are meaningful *relative* to each other on the same run, not as absolute model specs.
- **Memory is best-effort.** It uses the runner's resident size where exposed (Ollama `/api/ps`, LM Studio `/api/v0`) plus RSS sampling; on unified-memory Macs it's approximate, and client-timed for OpenAI-compatible backends.
- **`fit` sizes are estimates** (weights + KV cache + overhead) — treat "fits/tight" as guidance, not a guarantee. HuggingFace param counts come from safetensors metadata, which is missing for GGUF-only or gated repos.
- **Throughput scaling only appears on batching servers** (vLLM, etc.); a single local model serializes requests.

## Development

```bash
git clone https://github.com/david-g-3654/homebench
cd homebench
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

The codebase is small and layered: `providers/` (pluggable backends), `quality/` (tasks, graders, judge), `metrics/` (memory sampling), `runner.py` (orchestration), `report.py` (export + tables), and `tui/` + `plainui.py` (rendering). Adding a provider means subclassing `Provider` (or `OpenAICompatibleProvider`) and registering it; adding a task means appending to the suite in `quality/tasks.py` with a reference that satisfies its grader (enforced by the tests).

Contributions welcome — new providers, task packs, and metrics especially.

## Roadmap

- [x] PyPI release
- [x] HTML / shareable report export
- [x] Per-run environment capture (OS, RAM, GPU) for comparable results
- [ ] A composite "best model for your laptop" value score
- [ ] `homebench doctor` — diagnose provider / models / setup
- [ ] MLX provider (Apple-Silicon-native)
- [ ] Regression guard for CI (`diff --fail-on-regression`)
- [ ] Community task-pack sharing

## License

MIT
