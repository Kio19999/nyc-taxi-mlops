.PHONY: notebooks flow flow-serve prefect-ui setup lint format test data features train export serve batch docker-build docker-run pipeline up down mlflow-ui

setup:            ## install deps + git hooks
	uv sync
	uv run pre-commit install

lint:
	uv run ruff check src tests pipelines scripts

format:
	uv run ruff format src tests pipelines scripts

test:
	uv run pytest

data:             ## download raw TLC files
	uv run python -m taxi_demand.data.ingest

features:         ## clean + validate + aggregate + build features
	uv run python -m taxi_demand.features.build

train:            ## baselines + LightGBM, logged to MLflow
	uv run python -m taxi_demand.training.train

mlflow-ui:        ## view runs without docker (http://localhost:5001)
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

export:           ## export current champion -> models/champion/
	uv run python -m taxi_demand.serving.export

serve:            ## run the API locally (http://localhost:8000/docs)
	uv run uvicorn taxi_demand.serving.app:app --reload --port 8000

batch:            ## forecast all zones for the next hour
	uv run python -m taxi_demand.serving.batch

docker-build:
	docker build -f docker/api.Dockerfile -t taxi-demand-api:latest .

docker-run:
	docker run --rm -p 8000:8000 taxi-demand-api:latest

flow:             ## run the whole pipeline once as a Prefect flow
	uv run python pipelines/training_flow.py

flow-serve:       ## schedule it monthly (keep this terminal open); view runs in prefect-ui
	uv run python pipelines/training_flow.py --serve

prefect-ui:       ## Prefect dashboard at http://localhost:4200
	PREFECT_SERVER_ANALYTICS_ENABLED=false uv run prefect server start

notebooks:        ## open the analysis notebooks in JupyterLab
	uv run jupyter lab notebooks/

pipeline: data features train export

up:               ## start local MLflow + Postgres + MinIO
	docker compose up -d --build

down:
	docker compose down
