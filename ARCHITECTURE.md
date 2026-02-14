# Architecture — Sentiment Analysis MLOps Service

## 1. Overview

This project implements an end‑to‑end **Sentiment Analysis MLOps pipeline** that covers:

* Data ingestion and preprocessing
* Model training and experiment tracking (MLflow)
* Model registry and versioning
* FastAPI inference service
* Dockerized deployment
* Automated testing

The system follows a **production‑oriented ML service architecture** where training and inference are decoupled but share common artifacts via MLflow.

---

## 2. High‑Level Architecture

```
                ┌────────────────────┐
                │   IMDb Dataset     │
                │   (data/imdb.csv)  │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │   Data Loader      │
                │   src/data.py      │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │  Preprocessing     │
                │ src/preprocess.py  │
                │ TF‑IDF Vectorizer  │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Model Training     │
                │  src/train.py      │
                │  Logistic Reg      │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │  MLflow Tracking   │
                │  + Artifacts       │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │  MLflow Registry   │
                │  Production Model  │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ FastAPI Service    │
                │   app/main.py      │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │   REST Clients     │
                │ curl / Postman / UI│
                └────────────────────┘
```

---

## 3. Training Pipeline Architecture

### Steps

1. **Load dataset**

   * Source: `data/imdb.csv`
   * Module: `src/data.py`

2. **Text preprocessing**

   * Cleaning + TF‑IDF vectorization
   * Module: `src/preprocess.py`

3. **Model training**

   * Algorithm: Logistic Regression
   * Module: `src/train.py`

4. **Experiment tracking**

   * Metrics: accuracy, F1, etc.
   * Stored in: `mlruns/`

5. **Model logging**

   * MLflow model artifact
   * TF‑IDF vectorizer artifact

6. **Model registration**

   * Name: `SentimentClassifier`
   * Stage: Production
   * Module: `src/register_model.py`

---

## 4. Inference Architecture (FastAPI)

### Startup Phase

When FastAPI starts:

```
app startup
   ↓
Load MLflow client
   ↓
Fetch latest Production model
   ↓
Download TF‑IDF artifact
   ↓
Load vectorizer + model
```

### Prediction Flow

```
Client request
   ↓
POST /predict
   ↓
Pydantic validation
   ↓
TF‑IDF transform
   ↓
ML model predict
   ↓
Return sentiment + confidence
```

---

## 5. MLflow Integration Architecture

MLflow is used for:

* Experiment tracking
* Model artifact storage
* Model registry
* Version control
* Stage transitions

Artifacts stored:

```
mlruns/
 ├── experiments
 ├── runs
 ├── models
 └── tfidf_vectorizer/
```

FastAPI always loads:

```
models:/SentimentClassifier/Production
```

This guarantees:

* Versioned deployment
* Rollback capability
* Training/inference decoupling

---

## 6. Docker Architecture

### Services

1. **mlflow-server**

   * Port: 5000
   * Backend store: SQLite
   * Artifact store: local volume

2. **api**

   * FastAPI app
   * Port: 8000
   * Loads model from MLflow

### Container Communication

```
api container
   │
   │ MLFLOW_TRACKING_URI
   ▼
mlflow-server container
```

---

## 7. Project Structure Architecture Mapping

```
app/
  main.py              → Inference API

src/
  config.py            → Global paths & constants
  data.py              → Dataset loader
  preprocess.py        → Text preprocessing
  train.py             → Training pipeline
  register_model.py    → MLflow registration
  schemas.py           → Data schemas

data/
  imdb.csv             → Dataset

mlruns/
  → MLflow experiments & artifacts

tests/
  test_api.py          → API tests
  test_preprocess.py   → Preprocess tests

Dockerfile            → API container

docker-compose.yml    → Full system
```

---

## 8. Production Design Principles Applied

This project follows real MLOps best practices:

* Separation of training & inference
* Artifact‑based deployment
* Registry‑driven serving
* Containerized services
* Config‑driven paths
* Reproducible experiments
* Versioned models
* Automated tests

---

## 9. Deployment Flow (Production)

```
Train model
   ↓
Log to MLflow
   ↓
Register model
   ↓
Promote to Production
   ↓
Deploy API container
   ↓
API loads Production model
   ↓
Clients call API
```

---

## 10. Scalability & Future Extensions

Architecture supports:

* Replace TF‑IDF with BERT
* Replace Logistic Regression with DL model
* Remote artifact store (S3/GCS)
* Remote DB (Postgres)
* Kubernetes deployment
* CI/CD automation
* Batch inference pipeline
* Streaming inference

---

## 11. Key Architectural Strengths

* Registry‑based serving (industry standard)
* Version‑safe deployment
* Clear module separation
* Docker reproducibility
* Testable components
* Replaceable ML pipeline stages

---

## 12. End‑to‑End System Summary

```
Dataset → Preprocess → Train → MLflow → Registry → FastAPI → Client
```

This architecture matches modern production ML systems used in real companies.
