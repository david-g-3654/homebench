"""A curated catalog of popular local models + a hardware-fit calculator.

Sizes are *estimates*: GGUF weight size scales with parameter count and
quantization, plus a runtime overhead for the KV cache and the process. Real
usage varies with context length and backend, so treat "fits" as guidance,
not a guarantee.

Each model lists where to get it — an Ollama tag (`ollama pull …`) and the
HuggingFace repo (which is also what LM Studio and vLLM pull from).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

GB = 1_000_000_000

# Approximate GGUF bytes-per-billion-parameters at each quantization.
QUANT_GB_PER_B: Dict[str, float] = {
    "Q4_K_M": 0.60,
    "Q5_K_M": 0.72,
    "Q6_K": 0.82,
    "Q8_0": 1.10,
    "FP16": 2.00,
}
# Highest quality first — the fit search prefers the best quant that still fits.
QUANT_ORDER: List[str] = ["FP16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"]

# Fixed process/runtime overhead on top of weights + KV cache.
_OVERHEAD_BYTES = 0.8 * GB


@dataclass
class ModelSpec:
    name: str
    params_b: float
    family: str
    ollama: Optional[str] = None
    hf: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Spans tiny -> huge so the fit table is informative on any machine.
CATALOG: List[ModelSpec] = [
    ModelSpec("Qwen2.5 0.5B", 0.5, "qwen2", "qwen2.5:0.5b", "Qwen/Qwen2.5-0.5B-Instruct"),
    ModelSpec("Llama 3.2 1B", 1.24, "llama", "llama3.2:1b", "meta-llama/Llama-3.2-1B-Instruct"),
    ModelSpec("Gemma 2 2B", 2.6, "gemma2", "gemma2:2b", "google/gemma-2-2b-it"),
    ModelSpec("Llama 3.2 3B", 3.2, "llama", "llama3.2:3b", "meta-llama/Llama-3.2-3B-Instruct"),
    ModelSpec("Qwen2.5 3B", 3.1, "qwen2", "qwen2.5:3b", "Qwen/Qwen2.5-3B-Instruct"),
    ModelSpec("Phi-3.5 Mini", 3.8, "phi3", "phi3.5", "microsoft/Phi-3.5-mini-instruct"),
    ModelSpec("Mistral 7B", 7.2, "mistral", "mistral:7b", "mistralai/Mistral-7B-Instruct-v0.3"),
    ModelSpec("Qwen2.5 7B", 7.6, "qwen2", "qwen2.5:7b", "Qwen/Qwen2.5-7B-Instruct"),
    ModelSpec("Qwen2.5-Coder 7B", 7.6, "qwen2", "qwen2.5-coder:7b", "Qwen/Qwen2.5-Coder-7B-Instruct"),
    ModelSpec("Llama 3.1 8B", 8.0, "llama", "llama3.1:8b", "meta-llama/Llama-3.1-8B-Instruct"),
    ModelSpec("DeepSeek-R1 Distill 7B", 7.6, "qwen2", "deepseek-r1:7b", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
    ModelSpec("Gemma 2 9B", 9.2, "gemma2", "gemma2:9b", "google/gemma-2-9b-it"),
    ModelSpec("Qwen2.5 14B", 14.8, "qwen2", "qwen2.5:14b", "Qwen/Qwen2.5-14B-Instruct"),
    ModelSpec("DeepSeek-R1 Distill 14B", 14.8, "qwen2", "deepseek-r1:14b", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"),
    ModelSpec("Gemma 2 27B", 27.2, "gemma2", "gemma2:27b", "google/gemma-2-27b-it"),
    ModelSpec("Qwen2.5 32B", 32.5, "qwen2", "qwen2.5:32b", "Qwen/Qwen2.5-32B-Instruct"),
    ModelSpec("Qwen2.5-Coder 32B", 32.5, "qwen2", "qwen2.5-coder:32b", "Qwen/Qwen2.5-Coder-32B-Instruct"),
    ModelSpec("Mixtral 8x7B", 46.7, "mixtral", "mixtral:8x7b", "mistralai/Mixtral-8x7B-Instruct-v0.1"),
    ModelSpec("Llama 3.3 70B", 70.0, "llama", "llama3.3:70b", "meta-llama/Llama-3.3-70B-Instruct"),
]


# ---- estimation ------------------------------------------------------------
def weight_bytes(params_b: float, quant: str) -> int:
    return int(params_b * QUANT_GB_PER_B[quant] * GB)


def kv_cache_bytes(params_b: float, context: int) -> int:
    """Very rough KV-cache estimate, scaled by params and context length."""
    per_4k = 0.10 * params_b * GB          # ~0.1 GB per 1B params at 4k ctx
    return int(per_4k * (context / 4096.0))


def required_bytes(params_b: float, quant: str, context: int = 4096) -> int:
    return int(
        weight_bytes(params_b, quant)
        + kv_cache_bytes(params_b, context)
        + _OVERHEAD_BYTES
    )


# ---- fit --------------------------------------------------------------------
# status values, best -> worst
FIT = "fits"      # comfortable headroom
TIGHT = "tight"   # fits but little headroom
NO = "no"         # won't fit at this quant


def _status(required: int, budget: int) -> str:
    if budget <= 0:
        return NO
    ratio = required / budget
    if ratio <= 0.85:
        return FIT
    if ratio <= 1.0:
        return TIGHT
    return NO


@dataclass
class FitResult:
    model: ModelSpec
    quant: Optional[str]        # best quant that fits, or the smallest if none
    required_bytes: int
    status: str                 # FIT / TIGHT / NO

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model.to_dict(),
            "quant": self.quant,
            "required_bytes": self.required_bytes,
            "status": self.status,
        }


def best_fit(model: ModelSpec, budget: int, context: int = 4096,
             quant: Optional[str] = None) -> FitResult:
    """Pick the highest-quality quant that fits the budget.

    If a specific ``quant`` is given, evaluate only that one. If nothing fits,
    return the smallest quant with status NO.
    """
    quants = [quant] if quant else QUANT_ORDER
    smallest = None
    for q in quants:
        req = required_bytes(model.params_b, q, context)
        st = _status(req, budget)
        if st in (FIT, TIGHT):
            return FitResult(model, q, req, st)
        smallest = FitResult(model, q, req, NO)
    return smallest  # nothing fit; smallest quant evaluated last


def evaluate_catalog(budget: int, context: int = 4096,
                     quant: Optional[str] = None,
                     catalog: Optional[List[ModelSpec]] = None) -> List[FitResult]:
    models = catalog if catalog is not None else CATALOG
    return [best_fit(m, budget, context, quant) for m in models]
