# Computer Shop build evaluator

Build evaluator for Computer Shop, a web shop for PC parts. On the Build a PC page you fill eight slots (processor, motherboard, CPU cooler, memory, graphics card, storage, power supply and case), say what the PC is for (gaming, content creation or everyday work) and pick a resolution (1080p, 1440p or 4K). You get back a score from 0 to 100 and up to three concrete changes from the shop's own catalog. The score comes from a small gradient-boosted model trained offline on 4,000 LLM-labeled builds, exported to ONNX and served by a dedicated Lambda. The suggestions come from Amazon Nova Lite on Bedrock, asked on the fly by the same Lambda.

This repo holds the labeled data, the training pipeline, the trained `model.onnx` and the Lambda. The configurator that calls it lives in [computer_shop_ui](https://github.com/MartinSG98/computer_shop_ui) and posts the full product objects it already has loaded, so the Lambda never touches the database. The products come from [computer-shop-backend](https://github.com/MartinSG98/computer-shop-backend). Its `app/seed_data.py` is what the builds were labeled from, and `catalog.json` here is a snapshot of its `GET /products`. The Lambda, its model bucket and the `POST /evaluate` route on the shop's HTTP API are defined in [tf-module-computer_shop](https://github.com/MartinSG98/tf-module-computer_shop) and applied through [tf-stack-computer_shop](https://github.com/MartinSG98/tf-stack-computer_shop).

Architecture decisions are recorded in [docs/adr](docs/adr/README.md), one numbered file per decision.

## How a build is scored

1. **The request.** The configurator posts `{"build": {...}, "use_case": "gaming", "resolution": "1440p"}` to `/evaluate`. `build` has exactly the eight slots, keyed by category slug (`processors`, `motherboards`, `cpu-coolers`, `memory`, `graphics-cards`, `storage`, `power-supplies`, `cases`), and each holds the full catalog product with its `attributes`, `specs` and `price`. A body that is not JSON, a missing or extra slot, or an unknown use case or resolution answers 400 with the reason.
2. **Compatibility.** `compat.py`, a Python port of the configurator's compatibility engine (`src/lib/compatibility.ts` in computer_shop_ui), returns error and warning codes (see Compatibility rules).
3. **Features.** `features.py` turns the build and the scenario into 17 numbers. Training uses the same file, so the feature math cannot drift between training and serving.
4. **The score.** The ONNX model predicts a score, which is clipped to 0-100. A build with any compatibility error is capped at 19, so the gauge always reads broken, even when the model wobbles a little in the 0-20 range. The result is rounded to a whole number.
5. **Suggestions.** `suggest.py` asks Bedrock for up to three improvement tips (see Suggestions).
6. **The response.** Every response carries CORS headers for `POST` and `OPTIONS`, and an `OPTIONS` preflight answers 200 straight away.

```json
{"score": 88, "errors": [], "warnings": ["psu_headroom"], "recommendations": ["Upgrade to the ..."]}
```

The score says how well suited and balanced the build is for that use, following the labels:

- **Gaming** weighs the graphics card most, more so at higher resolutions, and favours CPUs with 3D V-Cache.
- **Content creation** weighs CPU cores and threads and RAM, plus graphics card VRAM.
- **Everyday** work suits modest parts. High-end parts count as overkill and lower the score.

Incompatible builds already score very low through the `error_count` feature and the labels, and the cap makes sure of it. With the current model, the flagship test build (Ryzen 9 9950X and RTX 5090) scores 93 for content creation at 4K and 23 for office work at 1080p.

## Compatibility rules

The draw estimate is the CPU's TDP plus the graphics card's TDP plus 100 W. Each rule skips itself when the parts or attributes it needs are missing.

| Code | Kind | When |
|---|---|---|
| `cpu_mobo_socket` | error | The CPU and motherboard sockets differ |
| `cooler_socket` | error | The cooler does not list the CPU's socket |
| `cooler_height` | error | An air cooler is taller than the case allows |
| `radiator_size` | error | An AIO radiator is larger than the case takes |
| `ram_type` | error | The memory type differs from the motherboard's |
| `ram_slots` | error | The kit has more sticks than the motherboard has slots |
| `mobo_form_factor` | error | The case does not take the motherboard's form factor |
| `gpu_length` | error | The graphics card is longer than the case allows |
| `psu_form_factor` | error | The case does not take the power supply's form factor |
| `psu_underpowered` | error | The power supply's wattage is below the estimated draw |
| `psu_headroom` | warning | The power supply covers the draw with less than 20 percent to spare |
| `cooler_underrated` | warning | The cooler's TDP rating is below the CPU's TDP |
| `ram_capacity` | warning | The kit is larger than the motherboard's maximum |
| `no_display` | warning | The CPU has no integrated graphics and there is no graphics card |

## Features

The order is fixed in `FEATURE_NAMES`, because training and inference have to agree on it.

| Feature | Made from |
|---|---|
| `cpu_cores`, `cpu_threads` | The first number in the CPU's `cores` and `threads` specs |
| `cpu_has_x3d` | 1 when the CPU's `cache` spec mentions 3D V-Cache |
| `cpu_tdp_w`, `cpu_tier` | The CPU's attributes |
| `gpu_vram_gb` | The first number in the graphics card's `vram` spec |
| `gpu_tdp_w`, `gpu_tier` | The graphics card's attributes |
| `ram_capacity_gb` | The memory kit's capacity |
| `est_draw_w` | The draw estimate above, 0 when neither TDP is known |
| `psu_headroom_ratio` | The power supply's wattage divided by the draw estimate |
| `error_count` | The number of compatibility errors |
| `total_price` | The sum of the eight prices |
| `use_gaming`, `use_content`, `use_everyday` | The use case, one-hot |
| `resolution` | 0 for 1080p, 1 for 1440p, 2 for 4k |

## Suggestions

After scoring, the Lambda calls the Bedrock Converse API with `SUGGEST_MODEL_ID`, which defaults to Amazon Nova Lite (`amazon.nova-lite-v1:0`), at temperature 0 and with at most 500 tokens. The message holds the use case, the resolution, the score, whether the build is compatible, a one-line summary of each part (its tier plus the specs that matter for its slot), and the same kind of summary for every buildable part in the bundled `catalog.json`, grouped by slot. That way the model has every part the shop sells in front of it, and the prompt forbids it to name anything else.

The system prompt in `suggest.py` defines an upgrade as a higher-tier part in the same slot and a cheaper option as a lower-tier one, says what matters for each use case, and makes every tip swap exactly one part for another from the catalog. Tips start with "Upgrade to ..." for a stronger part, or "Save money with ..." or "A cheaper option is ..." for a weaker one. A build scoring 90 or more gets at most one tip, and a build that is already a strong fit gets none. The reply has to be the JSON object `{"suggestions": [...]}`. The parser reads from the first `{` to the last `}`, which tolerates code fences, and keeps at most three non-empty tips.

Suggestions are best effort. Any failure, from missing Bedrock access to a timeout or a reply that does not parse, returns an empty list, so the score always comes back ([ADR 0002](docs/adr/0002-best-effort-bedrock-suggestions.md)). `boto3` comes with the Lambda runtime, so it is not in `app/requirements.txt`.

## Running locally

Requires Python 3.11 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
python tests/smoke_test.py
```

There is no local server. The handler is a plain function, so the tests call it in-process with simulated API Gateway events, and with `MODEL_BUCKET` unset it loads `model.onnx` from the repo root. `requirements.txt` holds the training packages (numpy, scikit-learn, skl2onnx, onnx and onnxruntime), which also cover the handler. Suggestions need boto3 as well (`pip install boto3`), plus AWS credentials with Bedrock access to Nova Lite and a default region where it is available, such as eu-west-2, set with `AWS_DEFAULT_REGION` or in your AWS profile. Without those the score still works and `recommendations` comes back empty.

## Configuration

These are the Lambda's environment variables. Terraform sets all four in AWS.

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_BUCKET` | unset | S3 bucket the model is loaded from. Unset loads `model.onnx` from the repo root |
| `MODEL_KEY` | `model.onnx` | Key of the model in that bucket |
| `ALLOWED_ORIGIN` | `*` | `Access-Control-Allow-Origin` on every response |
| `SUGGEST_MODEL_ID` | `amazon.nova-lite-v1:0` | Bedrock model for the suggestions |

The model is loaded once per warm container and kept for every request after that.

## Training a model

```bash
python validate.py data/builds.jsonl   # writes data/clean.jsonl and prints a coverage report
python train.py                        # writes model.onnx
python tests/smoke_test.py             # eyeball the scores of sample builds
```

1. **Label builds.** An LLM (Claude or ChatGPT) generates and rates builds from the backend's seed file, with the prompt below, and the output goes to `data/builds.jsonl`, the dataset of record. It holds 4,000 builds, split almost evenly across the three use cases and the three resolutions.
2. **Validate.** `validate.py` drops rows that are not valid JSON, do not have exactly the eight slots, use an unknown product id or put a product in the wrong slot, or have a score outside 0-100 or an unknown use case or resolution. It keeps the first of any identical build and scenario rows. It takes several files at once and also accepts a JSON array written one object per line. The report counts rows per use case, per resolution and per 20-point score band. `data/clean.jsonl` is generated, so it is git-ignored. All 4,000 current rows pass.
3. **Train.** `train.py` resolves each row's product ids through `catalog.json`, builds the features, holds out 15 percent and trains a scikit-learn `GradientBoostingRegressor` (400 trees, learning rate 0.05, depth 3, subsample 0.9, seed 42). It prints the holdout MAE and R2 and the six most important features, then refits on every row and exports `model.onnx` with skl2onnx (opset 17, one float input of 17 features).
4. **Check.** `tests/smoke_test.py` prints the score of eleven hand-picked scenarios next to what you would expect.
5. **Ship.** Commit `model.onnx` and push to `main`. The deploy workflow uploads it to S3, where the Lambda loads it from.

### Data format

One JSON object per line:

```json
{"build": {"processors": "<id>", "motherboards": "<id>", "cpu-coolers": "<id>", "memory": "<id>", "graphics-cards": "<id>", "storage": "<id>", "power-supplies": "<id>", "cases": "<id>"}, "use_case": "gaming", "resolution": "1440p", "score": 88, "reason": "..."}
```

`use_case` is one of `gaming`, `content` or `everyday`, and `resolution` one of `1080p`, `1440p` or `4k`.

### Generating labeled data

Paste this prompt into an LLM along with the backend's seed file (`app/seed_data.py`), which lists every product and its id. Run it on one model for consistency, because mixing labelers with different scales adds noise. A single reply will not fit 2,000 lines, so the output comes in chunks: say "continue" and append each chunk to `data/builds.jsonl`.

```
Using ONLY the products in the attached file, generate and rate as many diverse
complete PC builds as you can, up to 2000, as JSONL (one JSON object per line).

Use only these 8 categories: processors, motherboards, cpu-coolers, memory,
graphics-cards, storage, power-supplies, cases.

Rules:
- Fill all 8 slots, one product each. Use product "id" values EXACTLY as written.
- Make builds diverse: budget to flagship; balanced AND bottlenecked; overkill;
  and genuinely INCOMPATIBLE ones (wrong socket, DDR5 RAM on a DDR4 board, GPU
  too long for the case, ATX board in an ITX case, PSU below the build's draw,
  SFX-only case with an ATX PSU). Spread evenly across all 3 use cases and 3
  resolutions. No duplicate builds.

Scoring (0 to 100, for the given use_case and resolution):
- Compatibility gate: if any parts don't work together, score under 20.
- Gaming: weight the GPU heavily, more at higher resolution; favor 3D V-Cache
  CPUs; PSU must cover the GPU comfortably.
- Content creation: weight CPU cores/threads and RAM; favor high core-count CPUs
  over gaming-only chips; GPU VRAM matters.
- Everyday/office: modest parts are ideal; high-end parts are overkill and lower
  the score.
- Bands: 90-100 excellent, 70-89 solid, 50-69 works but imbalanced/poor value,
  30-49 serious problems, 0-29 incompatible.

Output ONLY JSONL, one object per line, no prose or code fences:
{"build": {"processors": "<id>", "motherboards": "<id>", "cpu-coolers": "<id>", "memory": "<id>", "graphics-cards": "<id>", "storage": "<id>", "power-supplies": "<id>", "cases": "<id>"}, "use_case": "gaming", "resolution": "1440p", "score": 88, "reason": "<one short sentence>"}
```

### Refreshing the catalog snapshot

`catalog.json` is used three times. Validation checks ids and slots against it, training resolves ids through it, and the Lambda bundles it for the suggestions' catalog summary. The score itself does not depend on it, since the configurator posts full products. When the shop's catalog changes, point `PRODUCTS_API` at the backend's base URL, refresh the snapshot and redeploy:

```bash
curl -s "$PRODUCTS_API/products" -o catalog.json
```

## Tests

```bash
python tests/test_handler.py
python tests/smoke_test.py
python tests/suggest_test.py
```

- `test_handler.py` runs the handler in-process against the local model. A good build scores 70 or more, an AM5 CPU on an Intel board is flagged `cpu_mobo_socket` and capped at 19, the same build scores higher for 4K gaming than for office work, a missing build or an unknown use case or resolution answers 400, and the preflight answers 200 with a CORS header. It prints PASS or FAIL per test and exits non-zero on a failure, and it also runs under pytest if you have it installed. If boto3, AWS credentials and a default region are available, every scored request also makes a live Bedrock call for suggestions.
- `smoke_test.py` prints eleven scenarios next to what you would expect, such as a flagship scoring low for office work and the 9800X3D beating the 9950X for gaming but not for content creation.
- `suggest_test.py` prints the live Bedrock suggestions for five scenarios, so you can check that every part they name exists in the catalog. It needs AWS credentials with `bedrock:InvokeModel` on Nova Lite and a default region where Nova Lite is available. When every scenario comes back empty, the cause is usually credentials, the region or model access, not five perfect builds.

Nothing runs these in CI. The only workflow is the deploy.

## Deployment

`.github/workflows/deploy.yml` runs on pushes to `main`, except ones that only touch Markdown files, and can also be started by hand.

1. **Package.** On Python 3.12, which matches the Lambda runtime on Amazon Linux 2023 so that the onnxruntime and numpy wheels load, it installs `app/requirements.txt` into a build folder and adds `features.py`, `compat.py`, `catalog.py`, `suggest.py`, `catalog.json` and `app/handler.py`.
2. **Authenticate.** It assumes the evaluator's deploy role through GitHub OIDC, so no AWS keys are stored.
3. **Upload the model.** `model.onnx` goes to the models bucket at `MODEL_KEY`, where the Lambda reads it.
4. **Deploy the code.** The zip is too large for a direct upload, so it goes to `build-eval/code.zip` in the same bucket, and `aws lambda update-function-code --publish` points the function at it.

The workflow reads four repository variables (Settings, Secrets and variables, Actions, Variables), each from a `terraform output` of [tf-stack-computer_shop](https://github.com/MartinSG98/tf-stack-computer_shop). It strips stray whitespace from them before use.

| Variable | Terraform output |
|---|---|
| `AWS_DEPLOY_ROLE_ARN` | `github_eval_deploy_role_arn` |
| `EVAL_LAMBDA_FUNCTION` | `eval_lambda_function_name` |
| `MODELS_BUCKET` | `models_bucket_name` |
| `MODEL_KEY` | `eval_model_key` |

Terraform creates the Lambda (Python 3.12, handler `app.handler.handler`), the versioned models bucket, the `POST /evaluate` and `OPTIONS /evaluate` routes and the deploy role before the first deploy, and it ignores code changes after that. The execution role can read the model and call `bedrock:InvokeModel` on the suggestion model. Without Bedrock access the score still works and the suggestions just come back empty. Deploys share a concurrency group, so a second push waits for the first instead of cancelling it.

## Structure

```
app/
├── handler.py          the Lambda: validation, compatibility, features, score, suggestions, CORS
└── requirements.txt    numpy and onnxruntime, the only packages in the Lambda zip
catalog.py              catalog.json as id to product, and the eight build slots
catalog.json            snapshot of the backend's GET /products, bundled into the Lambda
compat.py               Python port of the configurator's compatibility rules
features.py             build and scenario to the 17 features, shared by training and the Lambda
suggest.py              Bedrock prompt, catalog summary and reply parsing for the suggestions
validate.py             raw labeled rows to data/clean.jsonl, with a coverage report
train.py                trains the regressor and exports model.onnx
model.onnx              the trained model the deploy uploads to S3
data/builds.jsonl       the labeled builds, the dataset of record
tests/
├── test_handler.py     the handler with simulated API Gateway events
├── smoke_test.py       scores hand-picked builds to eyeball the model
└── suggest_test.py     prints live Bedrock suggestions for example builds
docs/adr/               architecture decision records
.github/workflows/
└── deploy.yml          package, upload the model and deploy the Lambda
requirements.txt        training and local dependencies
```
