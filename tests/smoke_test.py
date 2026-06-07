"""Readable sanity check of the trained evaluator.

Run from the repo root:  .venv/Scripts/python tests/smoke_test.py
(needs model.onnx + catalog.json in the repo root)

Scores a few hand-picked builds so you can eyeball that the model behaves:
good builds score high, incompatible ones score low, an over-the-top build for
office work scores low, and the same parts score differently per use case.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import onnxruntime as ort

from catalog import load_catalog, resolve_build
from compat import evaluate
from features import build_features

_catalog = load_catalog()
_session = ort.InferenceSession(os.path.join(ROOT, "model.onnx"))
_input = _session.get_inputs()[0].name


def score(build: dict, use_case: str, resolution: str) -> tuple[float, list[str]]:
    """Return (0-100 score, compatibility errors) for a build in a scenario."""
    resolved = resolve_build(build, _catalog)
    errors, _ = evaluate(resolved)
    features = np.array([build_features(resolved, use_case, resolution)], dtype=np.float32)
    raw = float(np.asarray(_session.run(None, {_input: features})[0]).ravel()[0])
    return round(max(0.0, min(100.0, raw)), 1), errors


# --- Example builds (slot -> product id). All compatible unless noted. ---

GOOD_GAMING = {
    "processors": "cpu-ryzen-9800x3d",
    "motherboards": "mobo-gigabyte-b650-elite-ax",
    "cpu-coolers": "cooler-thermalright-pa120-se",
    "memory": "ram-gskill-ddr5-6000-32gb",
    "graphics-cards": "gpu-gigabyte-rtx-5080",
    "storage": "ssd-samsung-990pro-1tb",
    "power-supplies": "psu-corsair-rm850e",
    "cases": "case-lian-li-216",
}

FLAGSHIP = {
    "processors": "cpu-ryzen-9950x",
    "motherboards": "mobo-gigabyte-x870e-master",
    "cpu-coolers": "cooler-arctic-lf3-420",
    "memory": "ram-gskill-ddr5-6400-96gb",
    "graphics-cards": "gpu-gigabyte-rtx-5090",
    "storage": "ssd-wd-sn850x-4tb",
    "power-supplies": "psu-seasonic-prime-tx1300",
    "cases": "case-lian-li-o11-xl",
}

MODEST_OFFICE = {
    "processors": "cpu-ryzen-9600x",
    "motherboards": "mobo-gigabyte-b650-elite-ax",
    "cpu-coolers": "cooler-thermalright-pa120-se",
    "memory": "ram-corsair-ddr5-5600-16gb",
    "graphics-cards": "gpu-msi-rtx-5050",
    "storage": "ssd-kingston-nv3-500gb",
    "power-supplies": "psu-bequiet-purepower12-550",
    "cases": "case-montech-air-903",
}

# Same as GOOD_GAMING but with an Intel board -> socket mismatch.
INCOMPATIBLE_SOCKET = dict(GOOD_GAMING, motherboards="mobo-gigabyte-z890-elite-wifi7")

# A 340mm GPU + ATX board + tall cooler crammed into an 11L ITX case.
INCOMPATIBLE_FIT = dict(GOOD_GAMING, cases="case-lian-li-a4-h2o")

# (label, build, use_case, resolution, what we expect)
CASES = [
    ("High-end gaming rig (9800X3D + 5080)", GOOD_GAMING, "gaming", "4k", "high"),
    ("Flagship for content creation", FLAGSHIP, "content", "4k", "high"),
    ("Flagship 9950X for gaming", FLAGSHIP, "gaming", "4k", "high-ish"),
    ("Flagship used for office work", FLAGSHIP, "everyday", "1080p", "low (overkill)"),
    ("Modest build for office work", MODEST_OFFICE, "everyday", "1080p", "good value"),
    ("Incompatible: AM5 CPU on Intel board", INCOMPATIBLE_SOCKET, "gaming", "1440p", "very low"),
    ("Incompatible: parts won't fit ITX case", INCOMPATIBLE_FIT, "gaming", "1440p", "very low"),
    # Same flagship parts, two uses -> the score should differ.
    ("9950X (16-core) for content", FLAGSHIP, "content", "4k", "compare below"),
    ("9800X3D (8-core) for content", dict(FLAGSHIP, processors="cpu-ryzen-9800x3d"), "content", "4k", "lower than 9950X"),
    ("9950X (16-core) for gaming", FLAGSHIP, "gaming", "4k", "compare below"),
    ("9800X3D (8-core) for gaming", dict(FLAGSHIP, processors="cpu-ryzen-9800x3d"), "gaming", "4k", "higher than 9950X"),
]


def main() -> None:
    print(f"{'build':<40}{'use case':<10}{'res':<7}{'score':>6}  expected")
    print("-" * 90)
    for label, build, use_case, resolution, expect in CASES:
        s, errors = score(build, use_case, resolution)
        flag = "  [incompatible]" if errors else ""
        print(f"{label:<40}{use_case:<10}{resolution:<7}{s:>6}  {expect}{flag}")


if __name__ == "__main__":
    main()