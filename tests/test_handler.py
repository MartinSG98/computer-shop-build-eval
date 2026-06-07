"""Tests for the Lambda handler (app/handler.py).

Runs the handler in-process with simulated API Gateway events, using the local
model.onnx fallback (no S3 needed). Run either way:

    .venv/Scripts/python tests/test_handler.py     # prints PASS/FAIL
    .venv/Scripts/python -m pytest tests/test_handler.py
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "app"))

from catalog import load_catalog  # noqa: E402
import handler  # noqa: E402

CATALOG = load_catalog()

GOOD = {
    "processors": "cpu-ryzen-9800x3d",
    "motherboards": "mobo-gigabyte-b650-elite-ax",
    "cpu-coolers": "cooler-thermalright-pa120-se",
    "memory": "ram-gskill-ddr5-6000-32gb",
    "graphics-cards": "gpu-gigabyte-rtx-5080",
    "storage": "ssd-samsung-990pro-1tb",
    "power-supplies": "psu-corsair-rm850e",
    "cases": "case-lian-li-216",
}
# AM5 CPU on an Intel board -> socket mismatch.
INCOMPATIBLE = dict(GOOD, motherboards="mobo-gigabyte-z890-elite-wifi7")


def _event(ids=None, use_case="gaming", resolution="4k", method="POST", raw=None) -> dict:
    if raw is None:
        build = {slot: CATALOG[pid] for slot, pid in ids.items()}
        raw = json.dumps({"build": build, "use_case": use_case, "resolution": resolution})
    return {"requestContext": {"http": {"method": method}}, "body": raw}


def _call(**kwargs) -> tuple[int, dict]:
    res = handler.handler(_event(**kwargs))
    return res["statusCode"], json.loads(res["body"])


def test_good_build_scores_high():
    status, body = _call(ids=GOOD, use_case="gaming", resolution="4k")
    assert status == 200
    assert body["errors"] == []
    assert body["score"] >= 70


def test_incompatible_build_is_capped_and_flagged():
    status, body = _call(ids=INCOMPATIBLE, use_case="gaming", resolution="4k")
    assert status == 200
    assert "cpu_mobo_socket" in body["errors"]
    assert body["score"] <= 19  # incompatible builds are forced low


def test_same_build_differs_by_use_case():
    _, gaming = _call(ids=GOOD, use_case="gaming", resolution="4k")
    _, everyday = _call(ids=GOOD, use_case="everyday", resolution="1080p")
    assert gaming["score"] > everyday["score"]  # high-end rig is overkill for office


def test_missing_build_returns_400():
    status, _ = _call(raw='{"use_case": "gaming", "resolution": "4k"}')
    assert status == 400


def test_bad_use_case_returns_400():
    status, _ = _call(ids=GOOD, use_case="mining", resolution="4k")
    assert status == 400


def test_bad_resolution_returns_400():
    status, _ = _call(ids=GOOD, use_case="gaming", resolution="8k")
    assert status == 400


def test_options_preflight_ok():
    res = handler.handler(_event(method="OPTIONS", raw="{}"))
    assert res["statusCode"] == 200
    assert res["headers"]["Access-Control-Allow-Origin"]


def _main() -> None:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}  {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    _main()