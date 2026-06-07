"""Validate and clean raw LLM-generated build rows into a training-ready file.

Usage:
    python validate.py raw1.jsonl raw2.jsonl ...   -> writes clean.jsonl + a report

Each input line should be:
    {"build": {slot: product_id, ... x8}, "use_case": "...", "resolution": "...",
     "score": 0-100, "reason": "..."}

Drops rows that: aren't valid JSON, use unknown product ids, put a product in the
wrong slot, miss a slot, or have an out-of-range score / bad use_case / resolution.
De-dupes identical build+scenario rows (keeps the first).
"""

import json
import sys
from collections import Counter

from catalog import BUILD_SLOTS, load_catalog
from features import RESOLUTION_ORDINAL, USE_CASES


def validate_row(row: dict, catalog: dict[str, dict]) -> str | None:
    """Return None if the row is valid, else a short reason string."""
    if not isinstance(row, dict):
        return "not_object"
    build = row.get("build")
    if not isinstance(build, dict):
        return "no_build"
    if set(build.keys()) != set(BUILD_SLOTS):
        return "wrong_slots"
    for slot, pid in build.items():
        product = catalog.get(pid)
        if product is None:
            return f"unknown_id:{pid}"
        if product["category"] != slot:
            return f"wrong_category:{pid}"
    if row.get("use_case") not in USE_CASES:
        return "bad_use_case"
    if row.get("resolution") not in RESOLUTION_ORDINAL:
        return "bad_resolution"
    score = row.get("score")
    if not isinstance(score, (int, float)) or not (0 <= score <= 100):
        return "bad_score"
    return None


def main(paths: list[str]) -> None:
    catalog = load_catalog()
    seen: set[str] = set()
    clean: list[dict] = []
    reasons: Counter = Counter()
    total = 0

    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip().rstrip(",")
                if not line or line in ("[", "]"):
                    continue
                total += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    reasons["bad_json"] += 1
                    continue
                reason = validate_row(row, catalog)
                if reason:
                    reasons[reason.split(":")[0]] += 1
                    continue
                key = json.dumps(row["build"], sort_keys=True) + row["use_case"] + row["resolution"]
                if key in seen:
                    reasons["duplicate"] += 1
                    continue
                seen.add(key)
                clean.append(row)

    with open("data/clean.jsonl", encoding="utf-8", mode="w") as f:
        for row in clean:
            f.write(json.dumps(row) + "\n")

    print(f"read {total} rows -> {len(clean)} valid written to data/clean.jsonl")
    if reasons:
        print("dropped:")
        for reason, count in reasons.most_common():
            print(f"  {count:5}  {reason}")
    # Coverage snapshot so we can see gaps before training.
    print("\ncoverage:")
    print("  use_case:  ", dict(Counter(r["use_case"] for r in clean)))
    print("  resolution:", dict(Counter(r["resolution"] for r in clean)))
    bands = Counter(min(int(r["score"]) // 20, 4) for r in clean)
    print("  score band:", {f"{b*20}-{b*20+19}": bands.get(b, 0) for b in range(5)})


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python validate.py <raw.jsonl> [more.jsonl ...]")
        sys.exit(1)
    main(sys.argv[1:])