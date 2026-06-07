"""Train the build-score model from clean.jsonl and export it to model.onnx.

Reads labeled rows (build + use_case + resolution + score), turns each into the
shared feature vector (features.py), trains a gradient-boosted regressor, reports
holdout accuracy, then refits on all data and exports to ONNX for the Lambda.
"""

import json

import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from catalog import load_catalog, resolve_build
from features import FEATURE_NAMES, build_features

CLEAN_PATH = "data/clean.jsonl"
MODEL_PATH = "model.onnx"


def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    catalog = load_catalog()
    rows = [json.loads(line) for line in open(CLEAN_PATH, encoding="utf-8") if line.strip()]
    X, y = [], []
    for row in rows:
        build = resolve_build(row["build"], catalog)
        X.append(build_features(build, row["use_case"], row["resolution"]))
        y.append(float(row["score"]))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def main() -> None:
    X, y = load_dataset()
    print(f"dataset: {X.shape[0]} rows x {X.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    model = GradientBoostingRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        random_state=42,
    )
    model.fit(X_train, y_train)

    pred = np.clip(model.predict(X_test), 0, 100)
    print(f"holdout MAE: {mean_absolute_error(y_test, pred):.1f} points")
    print(f"holdout R2:  {r2_score(y_test, pred):.3f}")

    # Most-important features, a quick sanity check on what the model leans on.
    importances = sorted(zip(FEATURE_NAMES, model.feature_importances_), key=lambda t: -t[1])
    print("top features:", [(n, round(v, 3)) for n, v in importances[:6]])

    # Refit on all data for the deployed model.
    model.fit(X, y)
    onnx_model = convert_sklearn(
        model,
        initial_types=[("input", FloatTensorType([None, X.shape[1]]))],
        target_opset=17,
    )
    with open(MODEL_PATH, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"wrote {MODEL_PATH}")


if __name__ == "__main__":
    main()