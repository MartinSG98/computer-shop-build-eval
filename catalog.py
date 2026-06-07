"""Load the product catalog snapshot (catalog.json) as an id -> product map.

The snapshot is taken from the live products API. Used at training time to
resolve a build's slot -> product-id map into full product objects. At
inference the Lambda receives the products in the request instead.
"""

import json
import os

_DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "catalog.json")

# The 8 buildable slots, in order. Keys match the product `category` slug.
BUILD_SLOTS = [
    "processors",
    "motherboards",
    "cpu-coolers",
    "memory",
    "graphics-cards",
    "storage",
    "power-supplies",
    "cases",
]


def load_catalog(path: str = _DEFAULT_PATH) -> dict[str, dict]:
    """Return {product_id: product} for every product in the snapshot."""
    with open(path, encoding="utf-8") as f:
        products = json.load(f)
    return {p["id"]: p for p in products}


def resolve_build(id_map: dict[str, str], catalog: dict[str, dict]) -> dict[str, dict]:
    """Turn a {slot: product_id} build into {slot: product}, skipping unknown ids."""
    resolved: dict[str, dict] = {}
    for slot, product_id in id_map.items():
        product = catalog.get(product_id)
        if product is not None:
            resolved[slot] = product
    return resolved