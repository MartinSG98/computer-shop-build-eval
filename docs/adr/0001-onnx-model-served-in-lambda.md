# ADR-0001: Offline-trained ONNX model served in Lambda

- Status: Accepted
- Date: 2026-07-09

## Context

The PC configurator needs to score a complete build 0-100 for a use case +
resolution. The score should be deterministic, fast, and cheap, and the exact
same feature math must be used at training and inference time.

## Decision

Train a gradient-boosted regressor offline on LLM-labeled builds, export to
ONNX, and serve it from a dedicated Lambda that loads `model.onnx` from S3 (with
a local-file fallback for dev). `features.py` is shared by training and the
Lambda so the math is identical. Incompatible builds are hard-capped low.

## Consequences

- Inference is fast and deterministic; no LLM in the scoring hot path.
- Retraining is an offline pipeline (validate -> train -> upload `model.onnx`);
  the Lambda just loads the latest model from S3.
- The shared `features.py` is the contract, so training and serving cannot drift.

## Alternatives considered

- Scoring with an LLM at request time: non-deterministic, slower, and more
  expensive per call. Rejected for the score (LLM is used only for suggestions,
  see ADR-0002).
- Rule-based heuristic scoring: hard to capture the "balanced / overkill" nuance
  the labels encode.
