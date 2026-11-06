# Fraud Detection: End-to-End MLOps Pipeline

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-tracking-0194E2?logo=mlflow&logoColor=white)
![DVC](https://img.shields.io/badge/DVC-versioned-13ADC7?logo=dvc&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-serving-009688?logo=fastapi&logoColor=white)
![Grafana](https://img.shields.io/badge/Grafana-monitoring-F46800?logo=grafana&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white)

An end-to-end MLOps pipeline for fraud detection on the IEEE-CIS dataset, built as my DDSC611 final-year project at ESLSCA University. It covers reproducible preprocessing, experiment tracking, a CI/CD data-validation gate that blocks training or deployment when schema and quality expectations are violated, containerised serving, and monitoring.

## Highlights

- **8 model families compared** with GridSearchCV and tracked in MLflow. The best model is registered as `FraudDetectionBestModel`.
- **Data-validation gate in CI.** Pandera schema checks, quality checks and drift tests (Kolmogorov-Smirnov, Jensen-Shannon) stop training or deployment on bad data.
- **Serving stack in one command.** FastAPI, a Gradio UI, MLflow, Prometheus and Grafana via Docker Compose.
- **Monitoring.** A provisioned Grafana dashboard plus Evidently baseline and drift reports.
- **Reproducible.** DVC-versioned data and models, Prefect orchestration, and GitHub Actions running Ruff, Black, Flake8 and unit tests.

## Results

Evaluated on a held-out test set of 20,000 transactions (2.56% fraud). The production model is selected by F1.

| Model | ROC-AUC | Avg. precision | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Gradient Boosting (selected)** | 0.911 | 0.681 | 0.814 | 0.580 | **0.677** |
| Hist Gradient Boosting | 0.917 | 0.683 | 0.827 | 0.553 | 0.663 |
| Random Forest | 0.912 | 0.716 | 0.934 | 0.496 | 0.648 |
| Extra Trees | 0.930 | 0.749 | 0.979 | 0.465 | 0.630 |
| Balanced Bagging | 0.893 | 0.617 | 0.740 | 0.539 | 0.624 |
| Balanced Random Forest | 0.931 | 0.669 | 0.859 | 0.475 | 0.611 |
| Easy Ensemble | 0.900 | 0.518 | 0.545 | 0.480 | 0.511 |
| RUSBoost | 0.909 | 0.555 | 0.218 | 0.752 | 0.338 |

Full details are in the [model card](docs/model_card.md) and [data card](docs/data_card.md).

<p align="center">
  <img src="reports/screenshots/MLFlow%20Runs.png" alt="MLflow experiment runs" width="49%">
  <img src="reports/screenshots/Evidently%20Drift%20Report.png" alt="Evidently drift report" width="49%">
</p>

## Quickstart

From a clean clone with Docker running, a reviewer can start the serving app and
monitoring stack in three commands:

```bash
python -m pip install -r requirements.txt
python -m dvc pull
docker compose up --build
```

After the services are healthy:

- FastAPI health: `http://localhost:8000/health`
- FastAPI prediction docs: `http://localhost:8000/docs`
- Gradio prediction UI: `http://localhost:7860`
- MLflow: `http://localhost:5000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` with username `admin` and password `admin`

## Architecture Overview

```text
data/
  raw/                 DVC-managed IEEE-CIS source files
  processed/           Prepared merged data
  splits/              Train/test splits
  reference/           Baseline validation statistics
reports/
  validation/          HTML and JSON validation outputs
logs/                  Rotating pipeline logs
configs/
  params.yaml          Training and preprocessing config
  validation.yaml      Data schema, quality, and drift thresholds
src/
  data/                Preprocessing utilities and feature engineering
  training/            MLflow model training and registration
  validation/          Typed config, IO, Pandera schema, checks, drift, reports
  pipeline/            CI/CD entrypoints
tests/
  data/                CI-safe sample and reference datasets
  unit/                Unit tests for preprocessing and validation
```

## Data Validation Gate

The validation system uses Pandera for schema enforcement and custom production
checks for quality and drift. It runs before training/deployment and returns a
non-zero exit code when any error-level check fails.

The validation package is split by responsibility:

- `config.py` validates YAML into typed runtime settings and applies path
  precedence as CLI argument, environment variable, then YAML config.
- `io.py` owns dataset loading and atomic artifact writes.
- `schema.py` builds the Pandera contract from YAML.
- `checks.py` exposes an extensible ordered quality-check suite.
- `drift.py` computes baseline statistics and distribution drift.
- `validator.py` orchestrates the run without owning low-level IO details.

Validation covers:

- Required columns, column names, data types, numeric ranges, and categorical
  allow-lists
- Missing value thresholds per column
- Duplicate business keys using `TransactionID`
- Invalid timestamps, future timestamps, and data freshness
- Outlier fractions using robust z-scores
- Unexpected categories and schema drift
- Numeric distribution drift with Kolmogorov-Smirnov tests
- Categorical distribution drift with Jensen-Shannon distance
- Class imbalance warnings for the target column

Reports are generated on every run:

- `reports/validation/validation_report.html`
- `reports/validation/validation_summary.json`
- `logs/data_validation.log`
- `data/reference/validation_baseline.json`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pre-commit install
```

Pull DVC artifacts when running the full fraud pipeline:

```bash
python -m dvc pull
```

Start the local serving and monitoring stack:

```bash
docker compose up serving-api gradio-ui prometheus grafana mlflow
```

Services:

- FastAPI serving API: `http://localhost:8000`
- FastAPI health endpoint: `http://localhost:8000/health`
- Prometheus metrics endpoint: `http://localhost:8000/metrics`
- Gradio prediction UI: `http://localhost:7860`
- MLflow tracking server: `http://localhost:5000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` with username `admin` and password `admin`

Inside Docker Compose, services communicate through Docker DNS names, for
example `http://mlflow:5000`, rather than `localhost`.

The Grafana dashboard is provisioned automatically from
`monitoring/grafana/dashboards/fraud-serving-dashboard.json`. It tracks API
availability, health, request rate, endpoint latency, prediction volume,
prediction errors, fraud probability, model confidence, feature completeness,
serving artifact status, and saved best-model evaluation metrics from
`reports/mlflow_experiment_results.csv`.

Generate the required Evidently monitoring reports:

```bash
python monitoring/run_monitoring.py
```

This writes:

- `monitoring/evidently_reports/baseline_report.html`
- `monitoring/evidently_reports/drift_report.html`
- `monitoring/evidently_reports/monitoring_summary.json`
- `logs/monitoring.log`

The baseline report compares a reference slice against a clean held-out slice.
The drift report compares the same reference slice against a deterministic
perturbed slice with drift injected into `TransactionAmt`, `TransactionDT`,
`card1`, `ProductCD`, `P_emaildomain`, and `card4`. If
`models/best_model.pkl` is present, the reports also include classification
performance metrics.

## Dockerized Serving

The serving stack is containerized with Docker Compose. The API container expects
the DVC-managed model and split artifacts to already exist on the host machine,
then mounts them read-only into the container.

Prerequisite:

```bash
python -m dvc pull
```

Confirm the required artifacts are present:

```bash
test -f models/best_model.pkl
test -f data/splits/test.csv
```

Start the serving stack:

```bash
docker compose up --build
```

Services:

- FastAPI serving API: `http://localhost:8000`
- FastAPI health endpoint: `http://localhost:8000/health`
- Gradio prediction UI: `http://localhost:7860`
- MLflow tracking server: `http://localhost:5000`

Inside Docker Compose, services communicate through Docker DNS names, for
example `http://mlflow:5000`, rather than `localhost`.

## Local Commands

Run the CI-safe validation gate:

```bash
python src/pipeline/validate_pipeline.py --config configs/validation.yaml
```

Validate a real processed dataset:

```bash
DATA_VALIDATION_INPUT_PATH=data/processed/cleaned_data.csv \
python src/pipeline/validate_pipeline.py --config configs/validation.yaml
```

Refresh baseline statistics from configured reference data:

```bash
python src/pipeline/validate_pipeline.py --config configs/validation.yaml --update-baseline
```

Run quality checks with Make:

```bash
make lint
make test
make validate
make ci
```

Run preprocessing and training:

```bash
python src/data/preprocessing.py
python src/training/train.py
python src/training/register_model.py
```

Reproduce the DVC pipeline:

```bash
dvc repro
```

## CI/CD

`.github/workflows/ci.yml` runs on pushes, pull requests, and manual dispatch.
The workflow stages are:

1. Install Python 3.11 dependencies
2. Run linting with Ruff, Black, and Flake8
3. Run unit tests with coverage
4. Run the data validation gate
5. Upload validation reports and logs as workflow artifacts

CI uses `tests/data/sample_fraud_transactions.csv` and
`tests/data/reference_fraud_transactions.csv` so the pipeline is executable
without downloading private DVC data. Production runs can override paths with:

- `DATA_VALIDATION_INPUT_PATH`
- `DATA_VALIDATION_REFERENCE_PATH`
- `DATA_VALIDATION_BASELINE_PATH`
- `DATA_VALIDATION_REPORT_DIR`
- `DATA_VALIDATION_LOG_DIR`

## Example Output

Successful CLI run:

```text
Data validation PASSED: 0 errors, 0 warnings.
HTML report: reports/validation/validation_report.html
JSON summary: reports/validation/validation_summary.json
```

Failed CLI run:

```text
Data validation FAILED: 2 errors, 1 warnings.
Top validation errors:
- [missing_value_threshold] TransactionAmt: Column 'TransactionAmt' has 15.00% missing values; allowed threshold is 0.00%.
```

## Report Screenshots

Add screenshots from a local or CI run to `reports/screenshots/`:

- Validation summary dashboard
- Error details table
- Drift metrics section
- GitHub Actions validation artifact

## Configuration

Edit `configs/validation.yaml` to tune thresholds without changing code. Common
production changes include:

- Lowering or raising null thresholds per feature
- Treating drift warnings as errors with `fail_on_distribution_drift`
- Switching `DATA_VALIDATION_INPUT_PATH` to the current training dataset
- Updating `reference_data_path` to the approved golden dataset
- Setting freshness to a real ingestion or event timestamp column

## Notes

- `data/raw/` is intentionally DVC-managed through `data/raw.dvc`.
- SMOTE remains inside the training pipeline so synthetic samples are created
  only within training folds.
- Validation runs before training/deployment to prevent bad data from entering
  MLflow experiments or model artifacts.
