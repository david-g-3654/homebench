# localbench

**Benchmark the local LLMs you already have — speed, memory, *and* quality — as a live terminal leaderboard.**

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

`localbench` is a single-command TUI that discovers the models installed in your local runner (**Ollama**, **LM Studio**, **llama.cpp**, **vLLM**, or any **OpenAI-compatible** server), runs a curated quality suite, measures **tokens/sec**, **time-to-first-token**, and **memory footprint** on *your actual machine*, and renders a live comparison leaderboard.

```bash
pipx install git+https://github.com/david-g-3654/localbench
localbench
```

That's it. No config, no API keys, no cloud. _(PyPI release — `pip install localbench` — is planned; see [Install](#install).)_

---

## Why

There are great tools for *one* half of this problem, but nothing local-first that does both:

- [`llama-bench`](https://github.com/ggml-org/llama.cpp) (inside llama.cpp) measures **speed only**.
- [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness) measures **quality** but has no polished laptop UX and isn't built around the model runners most people actually use locally.

`localbench` fills the gap: **local-first, zero-config, UX-driven.** Clone-and-run, point it at the models you already pulled, and get an at-a-glance answer to *"which of my local models is actually good, and how fast is it on this laptop?"*

## What it measures

| Metric | How |
| --- | --- |
| **tok/s** | Output tokens ÷ generation time. Ollama reports server-side eval timing; OpenAI-compatible backends are timed client-side from the token stream. Excludes prompt processing and model load. |
| **TTFT** | Wall-clock time to the first streamed token (minus model-load time where the runner reports it). |
| **Memory** | Resident model size when the runner exposes it (Ollama `/api/ps`, LM Studio `/api/v0`), plus a best-effort peak-RSS sample of the backend's processes. |
| **Quality** | 31 deterministically-graded tasks across math, reasoning, factual recall, instruction-following/structured-output, extraction, and code understanding. Optional **LLM-as-judge** adds open-ended tasks (summaries, email, haiku, explanations). |

## Install

Until a PyPI release lands, install from source:

```bash
# isolated, recommended
pipx install git+https://github.com/david-g-3654/localbench

# or clone and install
git clone https://github.com/david-g-3654/localbench
cd localbench
pip install .
```

Requires **Python 3.9+**.

## Usage

```bash
localbench                        # discover all models, run the full benchmark (TUI)
localbench --no-tui               # plain live renderer (great for piping / CI)
localbench -m llama3.2,qwen3:8b   # only these models
localbench --limit 3              # first 3 discovered models
localbench --provider lmstudio    # use LM Studio instead of auto-detect
localbench --provider llamacpp    # llama.cpp server (llama-server)
localbench --provider vllm        # vLLM
localbench --provider openai --host http://localhost:5000   # any OpenAI-compatible server
localbench --no-quality           # speed + memory only (fast)
localbench --no-speed             # quality only
localbench --judge qwen3:8b       # enable LLM-as-judge (adds open-ended tasks)
localbench --tasks mypack.yaml    # use a custom task pack instead of the built-in suite
localbench --add-tasks mypack.yaml  # add a pack on top of the built-in suite
localbench --label "before tuning"  # tag this run for later diffing
localbench --md results.md        # also export a Markdown report
localbench --json results.json    # also export raw JSON

localbench list                   # just list discovered models
localbench tasks                  # show the quality suite (add --tasks to preview a pack)
localbench history                # list past runs (saved automatically)
localbench diff                   # diff the two most recent runs
localbench diff 3 1               # diff run #3 (base) against run #1 (newer)
localbench throughput             # batch-throughput sweep (concurrency 1,2,4,8)
localbench throughput --concurrency 1,8,16 --provider vllm
```

Run `localbench --help` for the full flag list.

### Example output

```
                             Final leaderboard
┏━━━┳━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━┓
┃ # ┃ Model        ┃ Params ┃ Quality ┃  Pass ┃ tok/s ┃  TTFT ┃ Memory ┃
┡━━━╇━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━┩
│ 1 │ qwen3:8b     │   8.2B │    87%  │ 27/31 │  22.4 │ 210ms │ 5.2 GB │
│ 2 │ llama3.2     │   3.2B │    77%  │ 24/31 │  23.1 │ 150ms │ 2.4 GB │
│ 3 │ gemma3:4b    │   4.3B │    71%  │ 22/31 │  15.3 │ 360ms │ 3.5 GB │
└───┴──────────────┴────────┴─────────┴───────┴───────┴───────┴────────┘
```

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

The suite is small on purpose — enough tasks across categories to *separate* models, few enough that every model runs in a couple of minutes on a laptop. Each task is graded deterministically (exact numeric match, multiple-choice letter, substring, valid-JSON, regex). Temperature is 0 and a fixed seed is used for reproducibility. See `localbench tasks` for the list.

The optional `--judge MODEL` flag turns on an LLM-as-judge (any local model) that scores open-ended tasks 1–5 against a reference answer. It's a signal, not an oracle.

## Custom task packs

Bring your own evals with a JSON or YAML pack — no Python required. `--tasks` replaces the built-in suite; `--add-tasks` appends to it. YAML needs the optional extra (`pip install "localbench[yaml]"`); JSON works out of the box.

```yaml
# mypack.yaml  —  localbench --tasks mypack.yaml
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

Grader `type` values: `exact_number` (`value`, `tol`), `multiple_choice` (`value`), `contains_any` (`values`), `regex` (`pattern`, `ignorecase`), `valid_json` (`keys`), `valid_json_array` (`length`). Omit `grader` for a judge-only task. Runnable examples live in [`examples/`](examples/); preview any pack with `localbench tasks --tasks mypack.yaml`.

## History & diffing

Every run is saved automatically to `$LOCALBENCH_HOME/runs` (default `~/.localbench/runs`); disable with `--no-save`, and tag runs with `--label`.

```bash
localbench history            # table of past runs (newest first)
localbench diff               # previous run -> latest
localbench diff 3             # run #3 -> latest
localbench diff 3 1           # run #3 (base) -> run #1 (newer)
```

`diff` compares models by name and shows per-model deltas in quality and throughput, plus which models were added or removed between runs — handy for "did that quantization / setting actually help?"

## Batch throughput

The main leaderboard measures **single-stream** tok/s. Servers that batch requests (vLLM, llama.cpp continuous batching, Ollama with `OLLAMA_NUM_PARALLEL>1`) can do far more total work under concurrency — `localbench throughput` measures that:

```bash
localbench throughput -m my-model --concurrency 1,2,4,8
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

## Development

```bash
git clone https://github.com/david-g-3654/localbench
cd localbench
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

The codebase is small and layered: `providers/` (pluggable backends), `quality/` (tasks, graders, judge), `metrics/` (memory sampling), `runner.py` (orchestration), `report.py` (export + tables), and `tui/` + `plainui.py` (rendering). Adding a provider means subclassing `Provider` (or `OpenAICompatibleProvider`) and registering it; adding a task means appending to the suite in `quality/tasks.py` with a reference that satisfies its grader (enforced by the tests).

Contributions welcome — new providers, task packs, and metrics especially.

## Roadmap

- PyPI release
- HTML / shareable report export
- Per-run environment capture (OS, RAM, GPU) for comparable results

## License

MIT
