# localbench

**Benchmark the local LLMs you already have — speed, memory, *and* quality — as a live terminal leaderboard.**

`localbench` is a single-command TUI that discovers the models installed in your local runner (Ollama today), runs a small curated quality suite, measures **tokens/sec**, **time-to-first-token**, and **memory footprint** on *your actual machine*, and renders a live comparison leaderboard.

```
pip install localbench
localbench
```

That's it. No config, no API keys, no cloud.

---

## Why

There are great tools for *one* half of this problem, but nothing local-first that does both:

- [`llama-bench`](https://github.com/ggml-org/llama.cpp) (inside llama.cpp) measures **speed only**.
- [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness) measures **quality** but has no polished laptop UX and isn't built around the model runners most people actually use locally.

`localbench` fills the gap: **local-first, zero-config, UX-driven.** Clone-and-run, point it at the models you already pulled, and get an at-a-glance answer to *"which of my local models is actually good, and how fast is it on this laptop?"*

## What it measures

| Metric | How |
| --- | --- |
| **tok/s** | Output tokens ÷ generation time, reported by the runner (excludes prompt processing and model load). |
| **TTFT** | Wall-clock time to the first streamed token, minus model-load time. |
| **Memory** | Resident model size from the runner (`/api/ps`), plus a best-effort peak-RSS sample. |
| **Quality** | A curated suite across math, reasoning, factual recall, instruction-following, extraction, and code understanding — deterministically graded. Optional **LLM-as-judge** adds open-ended tasks. |

## Usage

```bash
localbench                      # discover all models, run the full benchmark (TUI)
localbench --no-tui             # plain live renderer (great for piping / CI)
localbench -m llama3.2,qwen3:8b # only these models
localbench --limit 3            # first 3 discovered models
localbench --no-quality         # speed + memory only (fast)
localbench --no-speed           # quality only
localbench --judge qwen3:8b     # enable LLM-as-judge (adds open-ended tasks)
localbench --md results.md      # also export a Markdown report
localbench --json results.json  # also export raw JSON

localbench list                 # just list discovered models
localbench tasks                # show the quality suite
```

Run `localbench --help` for the full flag list.

### Example output

```
                          Final leaderboard
┏━━━┳━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━┓
┃ # ┃ Model        ┃ Params ┃ Quality ┃ Pass ┃ tok/s ┃  TTFT ┃ Memory ┃
┡━━━╇━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━┩
│ 1 │ qwen3:8b     │   8.2B │    90%  │ 9/10 │  52.1 │ 210ms │ 5.2 GB │
│ 2 │ llama3.2     │   3.2B │    80%  │ 8/10 │  98.4 │  90ms │ 2.4 GB │
│ 3 │ gemma3:4b    │   4.3B │    70%  │ 7/10 │  74.9 │ 130ms │ 3.3 GB │
└───┴──────────────┴────────┴─────────┴──────┴───────┴───────┴────────┘
```

## Requirements

- Python 3.9+
- A local model runner. **Ollama** is supported today (`ollama serve` running, at least one model pulled).
  - Point at a non-default host with `--host` or the `OLLAMA_HOST` env var.

## How quality grading works

The suite is small on purpose — enough tasks across categories to *separate* models, few enough that every model runs in a couple of minutes on a laptop. Each task is graded deterministically (exact numeric match, multiple-choice letter, substring, valid-JSON, regex). Temperature is 0 and a fixed seed is used for reproducibility. See `localbench tasks` for the list.

The optional `--judge MODEL` flag turns on an LLM-as-judge (any local model) that scores open-ended tasks 1–5 against a reference answer. It's a signal, not an oracle.

## Roadmap

- More providers (LM Studio, llama.cpp server)
- Larger / pluggable task packs
- Historical runs & diffing

## License

MIT
