from homebench import catalog
from homebench.catalog import (
    CATALOG,
    FIT,
    NO,
    TIGHT,
    ModelSpec,
    best_fit,
    evaluate_catalog,
    required_bytes,
    weight_bytes,
)
from homebench.hardware import GPUInfo, HardwareInfo, capture
from homebench.report import fit_table, hardware_table

GB = 1_000_000_000


# ---- hardware --------------------------------------------------------------
def test_capture_returns_sane_values():
    hw = capture()
    assert hw.ram_total_bytes > 0
    assert hw.cpu_cores >= 1
    assert hw.os  # non-empty
    budget, label = hw.memory_budget()
    assert budget > 0 and isinstance(label, str)


def test_budget_prefers_nvidia_vram():
    hw = HardwareInfo(os="Linux", ram_total_bytes=64 * GB,
                      gpu=GPUInfo(name="RTX 4090", vram_bytes=24 * GB, kind="nvidia"))
    budget, label = hw.memory_budget()
    assert budget == int(24 * GB * 0.90)
    assert "VRAM" in label


def test_budget_apple_silicon_unified():
    hw = HardwareInfo(os="Darwin", arch="arm64", is_apple_silicon=True,
                      ram_total_bytes=32 * GB, gpu=GPUInfo(kind="apple"))
    budget, label = hw.memory_budget()
    assert budget == int(32 * GB * 0.72)
    assert "unified" in label.lower()


def test_budget_falls_back_to_ram():
    hw = HardwareInfo(os="Linux", ram_total_bytes=16 * GB)
    budget, label = hw.memory_budget()
    assert budget == int(16 * GB * 0.72)
    assert "RAM" in label


# ---- estimation ------------------------------------------------------------
def test_weight_and_required_scale():
    assert weight_bytes(8, "Q4_K_M") < weight_bytes(8, "Q8_0")
    assert weight_bytes(8, "Q8_0") < weight_bytes(8, "FP16")
    # required includes overhead, so it exceeds raw weights
    assert required_bytes(8, "Q4_K_M") > weight_bytes(8, "Q4_K_M")
    # bigger context -> more memory
    assert required_bytes(8, "Q4_K_M", 8192) > required_bytes(8, "Q4_K_M", 2048)


# ---- fit --------------------------------------------------------------------
def test_best_fit_picks_highest_quality_that_fits():
    m = ModelSpec("Test 7B", 7.0, "test", "test:7b", "org/test-7b")
    # ~48 GB budget -> FP16 (~14GB weights) fits comfortably
    r = best_fit(m, budget=48 * GB)
    assert r.status == FIT
    assert r.quant == "FP16"
    # ~6 GB budget -> only Q4 tier fits (tight)
    r2 = best_fit(m, budget=6 * GB)
    assert r2.quant == "Q4_K_M"
    assert r2.status in (FIT, TIGHT)


def test_best_fit_too_big_returns_no():
    m = ModelSpec("Huge 70B", 70.0, "test", "test:70b", "org/test-70b")
    r = best_fit(m, budget=4 * GB)
    assert r.status == NO
    assert r.quant is not None  # reports the smallest quant it tried


def test_forced_quant():
    m = ModelSpec("Test 7B", 7.0, "test")
    r = best_fit(m, budget=100 * GB, quant="Q4_K_M")
    assert r.quant == "Q4_K_M"  # honored even though FP16 would fit


def test_evaluate_catalog_shape():
    results = evaluate_catalog(budget=16 * GB)
    assert len(results) == len(CATALOG)
    # a tiny model must fit 16 GB; a 70B must not
    by_name = {r.model.name: r for r in results}
    assert by_name["Llama 3.2 1B"].status in (FIT, TIGHT)
    assert by_name["Llama 3.3 70B"].status == NO


# ---- catalog integrity -----------------------------------------------------
def test_catalog_integrity():
    names = [m.name for m in CATALOG]
    assert len(names) == len(set(names))          # unique names
    for m in CATALOG:
        assert m.params_b > 0
        assert m.ollama and m.hf                   # every model has get-it info


# ---- render ----------------------------------------------------------------
def test_render_helpers():
    hw = HardwareInfo(os="Darwin", arch="arm64", is_apple_silicon=True,
                      cpu="Apple M2", cpu_cores=8, ram_total_bytes=16 * GB,
                      ram_available_bytes=8 * GB, gpu=GPUInfo(name="Apple Silicon GPU",
                                                             kind="apple"))
    assert hardware_table(hw).row_count >= 4
    results = evaluate_catalog(budget=hw.memory_budget()[0])
    fitting = fit_table(results, show_all=False).row_count
    everything = fit_table(results, show_all=True).row_count
    assert everything == len(CATALOG)
    assert fitting <= everything
