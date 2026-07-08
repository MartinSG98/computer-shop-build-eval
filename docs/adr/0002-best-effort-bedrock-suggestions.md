# ADR-0002: Best-effort Bedrock suggestions alongside the score

- Status: Accepted
- Date: 2026-07-09

## Context

Beyond the numeric score, users benefit from a few concrete, catalog-grounded
tips on how to improve a build. That is qualitative and better suited to an LLM,
but it must never break the (deterministic) score response.

## Decision

After scoring, the same Lambda asks a Bedrock model (Amazon Nova Lite by default,
via the `SUGGEST_MODEL_ID` env var) for up to three improvement tips, passing the
build, its score, and a compact catalog summary so the model can only reference
real products. Suggestions are best-effort: any failure (no Bedrock access,
timeout, unparseable reply) is caught and returns an empty list.

## Consequences

- The score always responds even if suggestions fail; suggestions degrade to
  `[]`.
- Grounding on the catalog summary prevents invented products.
- Nova Lite keeps cost minimal, and the model is swappable via env var.

## Alternatives considered

- Baking suggestions into the ONNX model: not feasible; it outputs a scalar
  score.
- Making suggestions mandatory: would couple the score's availability to Bedrock.
  Rejected.
