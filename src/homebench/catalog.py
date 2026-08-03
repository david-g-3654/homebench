"""A curated catalog of popular local models + a hardware-fit calculator.

Sizes are *estimates*: GGUF weight size scales with parameter count and
quantization, plus a runtime overhead for the KV cache and the process. Real
usage varies with context length and backend, so treat "fits" as guidance,
not a guarantee.

Each model lists where to get it — an Ollama tag (`ollama pull …`) and the
HuggingFace repo (which is also what LM Studio and vLLM pull from).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple


class CatalogError(ValueError):
    """Raised when a user-supplied model catalog is malformed."""

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
    # --- tiny (<2B) ---
    ModelSpec("SmolLM2 135M", 0.135, "smollm2", "smollm2:135m", "HuggingFaceTB/SmolLM2-135M-Instruct"),
    ModelSpec("SmolLM2 360M", 0.36, "smollm2", "smollm2:360m", "HuggingFaceTB/SmolLM2-360M-Instruct"),
    ModelSpec("Qwen2.5 0.5B", 0.5, "qwen2", "qwen2.5:0.5b", "Qwen/Qwen2.5-0.5B-Instruct"),
    ModelSpec("TinyLlama 1.1B", 1.1, "llama", "tinyllama", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
    ModelSpec("Llama 3.2 1B", 1.24, "llama", "llama3.2:1b", "meta-llama/Llama-3.2-1B-Instruct"),
    ModelSpec("Qwen2.5 1.5B", 1.5, "qwen2", "qwen2.5:1.5b", "Qwen/Qwen2.5-1.5B-Instruct"),
    ModelSpec("SmolLM2 1.7B", 1.7, "smollm2", "smollm2:1.7b", "HuggingFaceTB/SmolLM2-1.7B-Instruct"),
    # --- small (2-4B) ---
    ModelSpec("Granite 3.1 2B", 2.5, "granite", "granite3.1-dense:2b", "ibm-granite/granite-3.1-2b-instruct"),
    ModelSpec("Gemma 2 2B", 2.6, "gemma2", "gemma2:2b", "google/gemma-2-2b-it"),
    ModelSpec("Qwen2.5 3B", 3.1, "qwen2", "qwen2.5:3b", "Qwen/Qwen2.5-3B-Instruct"),
    ModelSpec("Llama 3.2 3B", 3.2, "llama", "llama3.2:3b", "meta-llama/Llama-3.2-3B-Instruct"),
    ModelSpec("Qwen2.5-Coder 3B", 3.1, "qwen2", "qwen2.5-coder:3b", "Qwen/Qwen2.5-Coder-3B-Instruct"),
    ModelSpec("Phi-3.5 Mini", 3.8, "phi3", "phi3.5", "microsoft/Phi-3.5-mini-instruct"),
    # --- mid (6-9B) ---
    ModelSpec("CodeLlama 7B", 6.7, "llama", "codellama:7b", "codellama/CodeLlama-7b-Instruct-hf"),
    ModelSpec("Mistral 7B", 7.2, "mistral", "mistral:7b", "mistralai/Mistral-7B-Instruct-v0.3"),
    ModelSpec("Qwen2.5 7B", 7.6, "qwen2", "qwen2.5:7b", "Qwen/Qwen2.5-7B-Instruct"),
    ModelSpec("Qwen2.5-Coder 7B", 7.6, "qwen2", "qwen2.5-coder:7b", "Qwen/Qwen2.5-Coder-7B-Instruct"),
    ModelSpec("DeepSeek-R1 Distill 7B", 7.6, "qwen2", "deepseek-r1:7b", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
    ModelSpec("Llama 3.1 8B", 8.0, "llama", "llama3.1:8b", "meta-llama/Llama-3.1-8B-Instruct"),
    ModelSpec("DeepSeek-R1 Distill Llama 8B", 8.0, "llama", "deepseek-r1:8b", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"),
    ModelSpec("Granite 3.1 8B", 8.2, "granite", "granite3.1-dense:8b", "ibm-granite/granite-3.1-8b-instruct"),
    ModelSpec("Yi 1.5 9B", 8.8, "yi", "yi:9b", "01-ai/Yi-1.5-9B-Chat"),
    ModelSpec("Gemma 2 9B", 9.2, "gemma2", "gemma2:9b", "google/gemma-2-9b-it"),
    # --- large-ish (10-15B) ---
    ModelSpec("SOLAR 10.7B", 10.7, "llama", "solar", "upstage/SOLAR-10.7B-Instruct-v1.0"),
    ModelSpec("Mistral Nemo 12B", 12.2, "mistral", "mistral-nemo", "mistralai/Mistral-Nemo-Instruct-2407"),
    ModelSpec("CodeLlama 13B", 13.0, "llama", "codellama:13b", "codellama/CodeLlama-13b-Instruct-hf"),
    ModelSpec("Phi-3 Medium 14B", 14.0, "phi3", "phi3:14b", "microsoft/Phi-3-medium-4k-instruct"),
    ModelSpec("Qwen2.5 14B", 14.8, "qwen2", "qwen2.5:14b", "Qwen/Qwen2.5-14B-Instruct"),
    ModelSpec("Qwen2.5-Coder 14B", 14.8, "qwen2", "qwen2.5-coder:14b", "Qwen/Qwen2.5-Coder-14B-Instruct"),
    ModelSpec("DeepSeek-R1 Distill 14B", 14.8, "qwen2", "deepseek-r1:14b", "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"),
    ModelSpec("Phi-4 14.7B", 14.7, "phi3", "phi4", "microsoft/phi-4"),
    # --- big (20-35B) ---
    ModelSpec("Codestral 22B", 22.2, "mistral", "codestral", "mistralai/Codestral-22B-v0.1"),
    ModelSpec("Mistral Small 24B", 23.6, "mistral", "mistral-small", "mistralai/Mistral-Small-24B-Instruct-2501"),
    ModelSpec("Gemma 2 27B", 27.2, "gemma2", "gemma2:27b", "google/gemma-2-27b-it"),
    ModelSpec("Qwen2.5 32B", 32.5, "qwen2", "qwen2.5:32b", "Qwen/Qwen2.5-32B-Instruct"),
    ModelSpec("Qwen2.5-Coder 32B", 32.5, "qwen2", "qwen2.5-coder:32b", "Qwen/Qwen2.5-Coder-32B-Instruct"),
    ModelSpec("DeepSeek-R1 Distill 32B", 32.5, "qwen2", "deepseek-r1:32b", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"),
    ModelSpec("CodeLlama 34B", 33.7, "llama", "codellama:34b", "codellama/CodeLlama-34b-Instruct-hf"),
    ModelSpec("Yi 1.5 34B", 34.4, "yi", "yi:34b", "01-ai/Yi-1.5-34B-Chat"),
    ModelSpec("Command-R 35B", 35.0, "command-r", "command-r", "CohereForAI/c4ai-command-r-v01"),
    # --- huge (40-75B) ---
    ModelSpec("Mixtral 8x7B", 46.7, "mixtral", "mixtral:8x7b", "mistralai/Mixtral-8x7B-Instruct-v0.1"),
    ModelSpec("CodeLlama 70B", 69.0, "llama", "codellama:70b", "codellama/CodeLlama-70b-Instruct-hf"),
    ModelSpec("Llama 3.1 70B", 70.0, "llama", "llama3.1:70b", "meta-llama/Llama-3.1-70B-Instruct"),
    ModelSpec("Llama 3.3 70B", 70.0, "llama", "llama3.3:70b", "meta-llama/Llama-3.3-70B-Instruct"),
    ModelSpec("DeepSeek-R1 Distill Llama 70B", 70.0, "llama", "deepseek-r1:70b", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"),
    ModelSpec("Qwen2.5 72B", 72.7, "qwen2", "qwen2.5:72b", "Qwen/Qwen2.5-72B-Instruct"),
    # --- giant (100B+) ---
    ModelSpec("Command-R+ 104B", 104.0, "command-r", "command-r-plus", "CohereForAI/c4ai-command-r-plus"),
    ModelSpec("Mixtral 8x22B", 141.0, "mixtral", "mixtral:8x22b", "mistralai/Mixtral-8x22B-Instruct-v0.1"),
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


# ---- user-supplied catalogs ------------------------------------------------
def load_catalog(path: str) -> List[ModelSpec]:
    """Load extra models from a JSON file.

    Format: a list of objects (or ``{"models": [...]}``) with fields
    ``name`` and ``params_b`` (required), and optional ``family``, ``ollama``,
    ``hf``.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except OSError as exc:
        raise CatalogError(f"could not read catalog {path!r}: {exc}")
    except json.JSONDecodeError as exc:
        raise CatalogError(f"{path}: invalid JSON: {exc}")

    rows = data.get("models") if isinstance(data, dict) else data
    if not isinstance(rows, list) or not rows:
        raise CatalogError(f"{path}: expected a non-empty list of models")

    out: List[ModelSpec] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise CatalogError(f"{path}[{i}]: each model must be an object")
        if "name" not in row or "params_b" not in row:
            raise CatalogError(f"{path}[{i}]: 'name' and 'params_b' are required")
        try:
            params = float(row["params_b"])
        except (TypeError, ValueError):
            raise CatalogError(f"{path}[{i}]: 'params_b' must be a number")
        if params <= 0:
            raise CatalogError(f"{path}[{i}]: 'params_b' must be positive")
        out.append(ModelSpec(
            name=str(row["name"]),
            params_b=params,
            family=str(row.get("family", "custom")),
            ollama=row.get("ollama"),
            hf=row.get("hf"),
        ))
    return out
