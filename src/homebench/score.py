"""A composite "value" score: which model is the best fit for *this* laptop?

Combines quality, throughput, and memory into one 0–100 number, normalised
*within the run* (relative to the models you actually have), so the top model
is a concrete "best bang for your hardware" pick rather than an abstract rank.

Defaults lean on quality, then speed, then memory efficiency. Models missing a
metric (e.g. a ``--no-quality`` run) just have that weight redistributed.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .models import ModelReport

W_QUALITY = 0.5
W_SPEED = 0.3
W_MEMORY = 0.2


def _mem(r: ModelReport) -> int:
    return r.memory.size_bytes or r.model.size_bytes or 0


def value_scores(reports: List[ModelReport],
                 w_quality: float = W_QUALITY,
                 w_speed: float = W_SPEED,
                 w_memory: float = W_MEMORY) -> Dict[str, float]:
    """Return {model_name: value 0..100} for non-errored models."""
    ok = [r for r in reports if not r.error]
    if not ok:
        return {}
    max_q = max((r.quality_score for r in ok if r.quality_score is not None), default=0) or 0
    max_s = max((r.speed.tokens_per_sec for r in ok), default=0) or 0
    min_m = min((_mem(r) for r in ok if _mem(r)), default=0) or 0

    out: Dict[str, float] = {}
    for r in ok:
        num = 0.0
        wsum = 0.0
        if r.quality_score is not None and max_q > 0:
            num += w_quality * (r.quality_score / max_q)
            wsum += w_quality
        if max_s > 0:
            num += w_speed * (r.speed.tokens_per_sec / max_s)
            wsum += w_speed
        mem = _mem(r)
        if mem and min_m > 0:
            num += w_memory * (min_m / mem)  # smaller memory -> higher
            wsum += w_memory
        out[r.model.name] = round(100.0 * num / wsum, 1) if wsum > 0 else 0.0
    return out


def best_value(reports: List[ModelReport]) -> Tuple[Optional[ModelReport], Optional[float]]:
    scores = value_scores(reports)
    if not scores:
        return None, None
    name = max(scores, key=scores.get)
    rep = next((r for r in reports if r.model.name == name), None)
    return rep, scores.get(name)


def value_verdict(reports: List[ModelReport]) -> Optional[str]:
    """One-line 'best model for your laptop' headline, or None if not rankable."""
    rep, score = best_value(reports)
    if rep is None or len([r for r in reports if not r.error]) < 2:
        return None  # need at least two models for a "pick" to mean anything
    from .report import fmt_bytes, fmt_quality, fmt_tps  # local import avoids cycle

    bits = []
    if rep.quality_score is not None:
        bits.append(f"{fmt_quality(rep.quality_score)} quality")
    if rep.speed.tokens_per_sec:
        bits.append(f"{fmt_tps(rep.speed.tokens_per_sec)} tok/s")
    mem = _mem(rep)
    if mem:
        bits.append(fmt_bytes(mem))
    detail = ", ".join(bits)
    return f"{rep.model.name} — {detail} (value {score:g}/100)"
