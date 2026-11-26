# NYC Taxi Demand Forecasting — MLOps Platform

[![CI](https://github.com/Kio19999/nyc-taxi-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/Kio19999/nyc-taxi-mlops/actions/workflows/ci.yml)

Forecasts hourly taxi pickups for each of NYC's 263 taxi zones so a fleet can reposition
drivers ahead of demand. Built as a production-style ML system: versioned data, validated
pipelines, experiment tracking, gated model promotion, containerized serving, CI/CD,
drift monitoring and automated retraining on AWS.

## Architecture (status)
| Layer | Tooling | Status |
|---|---|---|
| Ingestion (monthly TLC Parquet) | requests, idempotent downloader | ✅ |
| Cleaning + quality gate | DuckDB SQL, drop-rate threshold | ✅ |
| Data contract | Pandera | ✅ |
| Features (lags, rolling, calendar) | pandas, leakage-safe | ✅ |
| Training + tracking | LightGBM (Poisson), baselines, MLflow | ✅ |
| Registry + champion/challenger gate | MLflow Model Registry (aliases) | ✅ |
| Serving (API + batch) | FastAPI, Docker (slim image) | ✅ — ECS in Phase 5 |
| Orchestration | Prefect flow (retries, monthly schedule, export only on promotion) | ✅ |
| CI/CD | GitHub Actions: lint, tests, Docker build + smoke test | ✅ |
| Infra | Terraform (S3, ECR, ECS, IAM) | ⏳ Phase 5 |
| Monitoring + retraining | CloudWatch, Evidently | ⏳ Phase 6 |

## Quickstart
```bash
make setup      # install deps + pre-commit hooks
make test       # run unit tests
make data       # download 2023-2024 yellow taxi data (~1.2 GB)
make features   # clean, validate, aggregate, build features
make train      # baselines + LightGBM -> MLflow -> promotion gate
make export     # champion -> models/champion/ serving bundle
make serve      # API at http://localhost:8000/docs
make flow       # whole pipeline as one Prefect flow (make prefect-ui for the dashboard)
make docker-build && make docker-run
```

## Results (test window Oct–Dec 2024, 263 zones)
| Model | WAPE | MAE |
|---|---|---|
| Naive (last hour) | 0.2625 | 4.83 |
| Seasonal naive (last week) | 0.2749 | 5.06 |
| **LightGBM** | **0.1481** | **2.72** |

LightGBM reduces WAPE by **43.6%** vs. the strongest baseline.

## Notebooks
- [`01_eda.ipynb`](notebooks/01_eda.ipynb): data quality, seasonality, zone concentration, autocorrelation (why lags 1h/24h/168h), year-over-year drift
- [`02_model_evaluation.ipynb`](notebooks/02_model_evaluation.ipynb): champion vs. baselines, error by hour and zone size, feature importance, bias

## Data
NYC TLC Trip Record Data (yellow taxi), aggregated to zone × hour pickup counts.
