# Sentiment Analysis MLOps Project

## Overview

This project is a production‑ready Sentiment Analysis system built with an end‑to‑end MLOps pipeline. It ingests data, preprocesses text, trains and tracks models with MLflow, registers the best model, and serves predictions through a FastAPI service. The system is containerized with Docker and orchestrated via Docker Compose for consistent local and deployment environments.

The design follows clean architecture principles and keeps training, inference, and serving layers modular while preserving your existing project structure and data flow.

---

## Key Features

* End‑to‑end ML pipeline (data → preprocess → train → register → serve)
* Experiment tracking and model registry with MLflow
* REST API for inference using FastAPI
* Reproducible environments with Docker & Docker Compose
* Config‑driven paths and parameters
* Unit tests for API and preprocessing
* Production‑oriented folder structure
* Model versioning and artifact storage

---

## Technology Stack

**Language & Core**

* Python 3.10+

**Machine Learning**

* scikit‑learn
* pandas
* numpy

**MLOps & Tracking**

* MLflow

**API & Serving**

* FastAPI
* Uvicorn / Gunicorn

**Containerization**

* Docker
* Docker Compose

**Testing**

* pytest

---

## Project Structure

```
app/
  main.py                # FastAPI app entrypoint

data/
  imdb.csv               # Raw dataset

mlruns/                  # MLflow artifacts & DB
  mlflow.db

picture/                 # Architecture & diagrams
  matrix.png
  modelinfo.png

src/
  config.py              # Central configuration
  data.py                # Data loading utilities
  preprocess.py          # Text preprocessing pipeline
  train.py               # Training + MLflow logging
  register_model.py      # Model registry logic
  schemas.py             # API request/response schemas

tests/
  test_api.py
  test_preprocess.py

.env.example             # Environment variables template
Dockerfile               # App container image

docker-compose.yml       # Multi‑service orchestration

requirements.txt         # Python dependencies
README.md
ARCHITECTURE.md
```

---

## Data & ML Pipeline Flow

1. **Data Load** → `src/data.py` reads dataset from `data/imdb.csv`
2. **Preprocess** → `src/preprocess.py` cleans & vectorizes text
3. **Train** → `src/train.py` trains model + logs to MLflow
4. **Register** → `src/register_model.py` promotes best model
5. **Serve** → `app/main.py` loads registered model and exposes API

Artifacts and metrics are stored in `mlruns/`.

---

## Environment Setup

### 1. Clone & Configure

```bash
git clone <repo>
cd <repo>
cp .env.example .env
```

Edit `.env` if needed (ports, MLflow URI, paths).

---

## Run with Docker (Recommended)

### Build & Start All Services

```bash
docker-compose up --build
```

Services:

* FastAPI → [http://localhost:8000](http://localhost:8000)
* API Docs → [http://localhost:8000/docs](http://localhost:8000/docs)
* MLflow → [http://localhost:5000](http://localhost:5000)

---

## Local Development (without Docker)

### Install

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

### Run MLflow

```bash
mlflow server \
  --backend-store-uri sqlite:///mlruns/mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5000
```

### Train Model

```bash
python -m src.train
```

### Run API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## API Usage

### Health Check

```
GET /
```

### Predict

```
POST /predict
{
  "text": "This movie was fantastic"
}
```

Response:

```
{
  "label": "positive",
  "confidence": 0.93
}
```

Interactive docs: `/docs`

---

## Testing

Run all tests:

```bash
pytest -v
```

Tests cover:

* API endpoints
* Preprocessing pipeline

---

## Production Deployment Notes

* Use Gunicorn with Uvicorn workers
* Persist `mlruns/` volume
* Set MLflow URI via env
* Disable reload
* Add reverse proxy (Nginx)

Example container CMD:

```bash
gunicorn -k uvicorn.workers.UvicornWorker app.main:app -b 0.0.0.0:8000
```

---

## Reproducibility & Versioning

* MLflow run tracking
* Registered model stages
* Docker image versioning
* Requirements lock

---

## Future Improvements

* CI/CD pipeline
* Data validation (Great Expectations)
* Drift monitoring
* Batch inference pipeline
* Cloud deployment (AWS/GCP/Azure)

---

## Author

Sadanand Kumar

BTech | Machine Learning Enthusiast
# sentiment-mlops-9
