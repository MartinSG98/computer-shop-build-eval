# computer-shop-build-eval

Build evaluator for the Computer Shop PC configurator. A small model scores a
complete PC build 0 to 100 for a given use case and resolution. Trained offline
on LLM-labeled builds, exported to ONNX, and served by a dedicated Lambda that
loads the model from S3.

## How a build is scored

Each build plus scenario (use case + resolution) becomes a fixed feature vector
(see `features.py`), and the model predicts a 0 to 100 score. The score reflects
how well-suited and balanced the build is for that use:

- Gaming: GPU-weighted (more at higher resolution), favors 3D V-Cache CPUs.
- Content creation: CPU cores/threads and RAM weighted, plus GPU VRAM.
- Everyday/office: modest parts ideal, high-end parts are overkill.

Incompatible builds score very low (the `error_count` feature plus the labels).

## Files

- `catalog.py` - load `catalog.json` (snapshot of the live products API) as id -> product; resolve a build's slot -> id map into products.
- `compat.py` - Python port of the frontend compatibility engine; returns errors/warnings for a build.
- `features.py` - build + scenario -> feature vector. **Shared by training and the Lambda** so the math is identical.
- `validate.py` - clean raw LLM rows into `clean.jsonl` (drops bad ids/slots/scores, dedupes, reports coverage).
- `train.py` - (next) read `clean.jsonl`, build features, train, export `model.onnx`.
- `app/handler.py` - (next) Lambda: load `model.onnx` from S3, request -> features -> score.

## Pipeline

1. Generate labeled builds with Claude/ChatGPT (build + use_case + resolution + score), save as raw `.jsonl`.
2. `python validate.py raw*.jsonl` -> `clean.jsonl` + a coverage report.
3. `python train.py` -> `model.onnx` (uploaded to S3).
4. Deploy the eval Lambda (Terraform, in the module) which loads the model from S3 and serves the score.

## Data format

One JSON object per line:

```json
{"build": {"processors": "<id>", "motherboards": "<id>", "cpu-coolers": "<id>", "memory": "<id>", "graphics-cards": "<id>", "storage": "<id>", "power-supplies": "<id>", "cases": "<id>"}, "use_case": "gaming", "resolution": "1440p", "score": 88, "reason": "..."}
```

`use_case`: gaming | content | everyday. `resolution`: 1080p | 1440p | 4k.

## Refreshing the catalog snapshot

```
curl -s https://api.msg-computers.com/products -o catalog.json
```