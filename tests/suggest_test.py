"""See what build suggestions Nova Lite produces (live Bedrock call).

Run from the repo root:  .venv/Scripts/python tests/suggest_test.py

Needs AWS credentials with bedrock:InvokeModel for amazon.nova-lite-v1:0 and a
region where Nova Lite is available. The region defaults to eu-west-2 below;
override with AWS_REGION if needed. Uses the same example builds as smoke_test.

Prints each build's score plus the model's improvement tips so you can eyeball
quality and grounding (every part it names should exist in the catalog). If
every scenario comes back empty, that usually means missing credentials, the
wrong region, or no Bedrock model access, not that the builds are perfect.
"""

import os
import sys

# Default the region before suggest.py builds its Bedrock client.
os.environ.setdefault("AWS_REGION", "eu-west-2")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from catalog import load_catalog, resolve_build
from smoke_test import FLAGSHIP, GOOD_GAMING, MODEST_OFFICE, score
from suggest import MODEL_ID, suggest

_catalog = load_catalog()

# (label, build, use_case, resolution)
CASES = [
    ("High-end gaming rig (9800X3D + 5080)", GOOD_GAMING, "gaming", "4k"),
    ("Same high-end rig used for office work", GOOD_GAMING, "everyday", "1080p"),
    ("Flagship for content creation", FLAGSHIP, "content", "4k"),
    ("Modest build for office work", MODEST_OFFICE, "everyday", "1080p"),
    ("Modest build pushed to 4k gaming", MODEST_OFFICE, "gaming", "4k"),
]


def main() -> None:
    print(f"Model: {MODEL_ID}  Region: {os.environ.get('AWS_REGION')}\n")
    for label, build, use_case, resolution in CASES:
        s, errors = score(build, use_case, resolution)
        resolved = resolve_build(build, _catalog)
        tips = suggest(resolved, use_case, resolution, round(s), not errors)

        header = f"{label}  |  {use_case} {resolution}  |  score {round(s)}"
        if errors:
            header += "  [incompatible]"
        print("=" * 72)
        print(header)
        if tips:
            for tip in tips:
                print("  -", tip)
        else:
            print("  (no suggestions: build is a good fit, or Bedrock returned nothing)")
    print("=" * 72)


if __name__ == "__main__":
    main()