"""Capture the host's environment and hardware.

Best-effort and cross-platform: RAM/CPU via psutil, GPU/VRAM via light probes
(``nvidia-smi`` when present; Apple Silicon treated as unified memory). Nothing
here raises — missing info just comes back empty/zero.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Tuple

import psutil


@dataclass
class GPUInfo:
    name: str = ""
    vram_bytes: int = 0
    kind: str = ""  # "nvidia" | "apple" | "amd" | ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HardwareInfo:
    os: str = ""
    os_version: str = ""
    arch: str = ""
    cpu: str = ""
    cpu_cores: int = 0
    ram_total_bytes: int = 0
    ram_available_bytes: int = 0
    is_apple_silicon: bool = False
    unified_memory: bool = False
    python_version: str = ""
    gpu: GPUInfo = field(default_factory=GPUInfo)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    # ---- memory budget -------------------------------------------------
    def memory_budget(self) -> Tuple[int, str]:
        """Usable memory (bytes) for loading a model, and a human label.

        - Discrete NVIDIA GPU: ~90% of VRAM (full-offload target).
        - Apple Silicon: ~72% of unified RAM (leave headroom for the OS/apps).
        - Otherwise: ~72% of system RAM (CPU / partial-offload).
        """
        if self.gpu.kind == "nvidia" and self.gpu.vram_bytes > 0:
            return int(self.gpu.vram_bytes * 0.90), f"GPU VRAM ({self.gpu.name})"
        if self.is_apple_silicon:
            return int(self.ram_total_bytes * 0.72), "Apple unified memory"
        return int(self.ram_total_bytes * 0.72), "system RAM"


# ---------------------------------------------------------------------------
def _run(cmd, timeout: float = 2.0) -> str:
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _mac_cpu_brand() -> str:
    return _run(["sysctl", "-n", "machdep.cpu.brand_string"])


def _detect_nvidia() -> GPUInfo:
    if not shutil.which("nvidia-smi"):
        return GPUInfo()
    out = _run([
        "nvidia-smi", "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ])
    if not out:
        return GPUInfo()
    # first GPU only
    first = out.splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    if len(parts) < 2:
        return GPUInfo()
    name = parts[0]
    try:
        vram_mib = float(parts[1])
    except ValueError:
        vram_mib = 0.0
    return GPUInfo(name=name, vram_bytes=int(vram_mib * 1024 * 1024), kind="nvidia")


def capture() -> HardwareInfo:
    system = platform.system()
    machine = platform.machine()

    vm = psutil.virtual_memory()
    cpu = platform.processor() or machine
    if system == "Darwin":
        cpu = _mac_cpu_brand() or cpu

    # arm64 is the obvious signal, but under an x86 Python running via Rosetta
    # machine() reports "x86_64" — fall back to the CPU brand ("Apple M…").
    is_apple_silicon = system == "Darwin" and (
        machine == "arm64" or cpu.startswith("Apple")
    )

    gpu = _detect_nvidia()
    if not gpu.kind and is_apple_silicon:
        gpu = GPUInfo(name="Apple Silicon GPU", vram_bytes=0, kind="apple")

    return HardwareInfo(
        os=system,
        os_version=platform.release(),
        arch=machine,
        cpu=cpu,
        cpu_cores=psutil.cpu_count(logical=True) or 0,
        ram_total_bytes=vm.total,
        ram_available_bytes=vm.available,
        is_apple_silicon=is_apple_silicon,
        unified_memory=is_apple_silicon,
        python_version=platform.python_version() or sys.version.split()[0],
        gpu=gpu,
    )
