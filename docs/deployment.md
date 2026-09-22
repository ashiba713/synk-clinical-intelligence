# SYNK Deployment Notes

Research prototype — not for clinical diagnosis or treatment.

## Local (recommended for research)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e . --no-deps

python -m scripts.cli generate-data
python -m scripts.cli train
python -m scripts.cli serve        # API  :8000
python -m scripts.cli dashboard    # UI   :8501
```

## Docker

```bash
docker compose up
# API  -> http://localhost:8000  (health at /health, docs at /docs)
# UI   -> http://localhost:8501
```

The image is CPU-only (python:3.11-slim + CPU torch). Data and artifacts live in named volumes
(`synk-data`, `synk-artifacts`); mount host directories instead to persist across rebuilds:

```yaml
volumes:
  - ./data:/app/data
  - ./artifacts:/app/artifacts
```

Training inside the container:

```bash
docker compose run --rm api python -m scripts.cli train --set training.max_epochs=5
```

## MLflow (optional)

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Then run training with:

```bash
SYNK_TRACKING__BACKEND=mlflow MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
    python -m scripts.cli train
```

Every run always writes a local JSON manifest under `artifacts/runs/<run_id>/manifest.json`
regardless of the tracking backend, so the platform remains fully usable offline.

## Environment variables

| Variable | Purpose |
| --- | --- |
| `SYNK_ENV` | development / research (no production mode exists) |
| `SYNK_LOG_LEVEL` | INFO / DEBUG / … |
| `SYNK_RANDOM_SEED` | global seed override |
| `SYNK_TEXT_ENCODER_TYPE` | `tfidf` (offline demo) / `transformer` |
| `SYNK_TEXT_MODEL_NAME` | HF model id, e.g. `emilyalsentzer/Bio_ClinicalBERT` |
| `DATABASE_URL` | SQLite (default) or PostgreSQL DSN |
| `MLFLOW_TRACKING_URI` | optional tracking backend |

See `.env.example` for the full list. Any config key can also be set via
`SYNK_SECTION__KEY=value` (e.g. `SYNK_MODEL__FUSION__TYPE=cross_attention`).

## Production caveats

- SYNK is **not** approved for clinical deployment; do not expose it to patient care networks.
- The bundled demo engine (`sklearn_demo`) is a transparent fallback for pre-training use; replace it
  with a registered torch bundle before any serious research serving.
- If authentication/multi-tenancy is ever needed, add it at the reverse-proxy layer; the API ships
  without auth by design for local research use.
