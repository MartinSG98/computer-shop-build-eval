"""AWS Lambda handler for the build evaluator.

POST a build (products per slot) + scenario; get back a 0-100 score.

Request body:
    {
      "build": { "processors": <product>, ... all 8 slots ... },
      "use_case": "gaming" | "content" | "everyday",
      "resolution": "1080p" | "1440p" | "4k"
    }
where each <product> is the full catalog object (with `attributes`, `specs`,
`price`). The frontend already has these loaded, so the Lambda stays stateless.

Response: { "score": int, "errors": [...], "warnings": [...] }

The model (model.onnx) is loaded once per warm container from S3 (MODEL_BUCKET /
MODEL_KEY). For local testing, if MODEL_BUCKET is unset it loads ./model.onnx.
"""

import json
import os

import numpy as np
import onnxruntime as ort

from catalog import BUILD_SLOTS
from compat import evaluate
from features import RESOLUTION_ORDINAL, USE_CASES, build_features

ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")
# Incompatible builds are forced below this so the gauge always reads "broken",
# regardless of small model wobble around the 0-20 range.
INCOMPATIBLE_CAP = 19

_session: ort.InferenceSession | None = None


def _get_session() -> ort.InferenceSession:
    """Load the ONNX model once per container (from S3 in Lambda, local file otherwise)."""
    global _session
    if _session is None:
        bucket = os.environ.get("MODEL_BUCKET")
        if bucket:
            import boto3

            key = os.environ.get("MODEL_KEY", "model.onnx")
            data = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
            _session = ort.InferenceSession(data)
        else:
            local = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model.onnx")
            _session = ort.InferenceSession(local)
    return _session


def _response(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": ALLOWED_ORIGIN,
            "Access-Control-Allow-Headers": "content-type",
            "Access-Control-Allow-Methods": "POST,OPTIONS",
        },
        "body": json.dumps(body),
    }


def handler(event: dict, context=None) -> dict:
    method = ((event.get("requestContext") or {}).get("http") or {}).get("method", "POST")
    if method == "OPTIONS":  # CORS preflight
        return _response(200, {})

    try:
        body = event.get("body") or "{}"
        if event.get("isBase64Encoded"):
            import base64

            body = base64.b64decode(body).decode()
        payload = json.loads(body)
    except (ValueError, TypeError):
        return _response(400, {"error": "invalid JSON body"})

    build = payload.get("build")
    use_case = payload.get("use_case")
    resolution = payload.get("resolution")

    if not isinstance(build, dict) or set(build.keys()) != set(BUILD_SLOTS):
        return _response(400, {"error": f"build must include exactly these slots: {BUILD_SLOTS}"})
    if use_case not in USE_CASES:
        return _response(400, {"error": f"use_case must be one of {list(USE_CASES)}"})
    if resolution not in RESOLUTION_ORDINAL:
        return _response(400, {"error": f"resolution must be one of {list(RESOLUTION_ORDINAL)}"})

    errors, warnings = evaluate(build)
    features = np.array([build_features(build, use_case, resolution)], dtype=np.float32)
    session = _get_session()
    raw = float(np.asarray(session.run(None, {session.get_inputs()[0].name: features})[0]).ravel()[0])
    score = max(0.0, min(100.0, raw))
    if errors:
        score = min(score, INCOMPATIBLE_CAP)

    return _response(200, {"score": round(score), "errors": errors, "warnings": warnings})