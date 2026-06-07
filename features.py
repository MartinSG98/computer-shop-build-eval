"""Turn a build + scenario into the numeric feature vector for the model.

Shared by training (train.py) and inference (app/handler.py) so the feature
math is identical in both. A build here is {slot: product}, where each product
is a catalog object with `attributes` and `specs`.
"""

import re

from compat import estimated_draw, evaluate

# Feature order is fixed: training and inference must agree on it.
FEATURE_NAMES = [
    "cpu_cores",
    "cpu_threads",
    "cpu_has_x3d",
    "cpu_tdp_w",
    "cpu_tier",
    "gpu_vram_gb",
    "gpu_tdp_w",
    "gpu_tier",
    "ram_capacity_gb",
    "est_draw_w",
    "psu_headroom_ratio",
    "error_count",
    "total_price",
    "use_gaming",
    "use_content",
    "use_everyday",
    "resolution",
]

USE_CASES = ("gaming", "content", "everyday")
RESOLUTION_ORDINAL = {"1080p": 0, "1440p": 1, "4k": 2}


def _leading_int(text: str | None) -> int:
    """First integer in a string, e.g. '24 (8P + 16E)' -> 24, '16GB GDDR7' -> 16."""
    if not text:
        return 0
    match = re.search(r"\d+", text)
    return int(match.group()) if match else 0


def _attrs(product: dict | None) -> dict:
    return (product or {}).get("attributes") or {}


def _specs(product: dict | None) -> dict:
    return (product or {}).get("specs") or {}


def build_features(build: dict[str, dict], use_case: str, resolution: str) -> list[float]:
    """Build + scenario -> feature vector aligned with FEATURE_NAMES."""
    cpu, cpu_a, cpu_s = build.get("processors"), _attrs(build.get("processors")), _specs(build.get("processors"))
    gpu_a, gpu_s = _attrs(build.get("graphics-cards")), _specs(build.get("graphics-cards"))
    ram_a = _attrs(build.get("memory"))
    psu_a = _attrs(build.get("power-supplies"))

    draw = estimated_draw(build)
    psu_w = psu_a.get("wattage_w") or 0
    errors, _ = evaluate(build)
    total_price = sum(float(p["price"]) for p in build.values() if p and p.get("price") is not None)

    return [
        float(_leading_int(cpu_s.get("cores"))),
        float(_leading_int(cpu_s.get("threads"))),
        1.0 if "3D V-Cache" in (cpu_s.get("cache") or "") else 0.0,
        float(cpu_a.get("tdp_w") or 0),
        float(cpu_a.get("tier") or 0),
        float(_leading_int(gpu_s.get("vram"))),
        float(gpu_a.get("tdp_w") or 0),
        float(gpu_a.get("tier") or 0),
        float(ram_a.get("capacity_gb") or 0),
        float(draw),
        float(psu_w / draw) if draw > 0 else 0.0,
        float(len(errors)),
        float(total_price),
        1.0 if use_case == "gaming" else 0.0,
        1.0 if use_case == "content" else 0.0,
        1.0 if use_case == "everyday" else 0.0,
        float(RESOLUTION_ORDINAL.get(resolution, 0)),
    ]