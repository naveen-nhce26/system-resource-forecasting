# Time Series Forecasting for System Resource Utilization

> An end-to-end time-series forecasting platform for predicting system resource utilization to support capacity planning, performance monitoring, and proactive infrastructure management.

**Python · Pandas · NumPy · Scikit-learn · SARIMA · Prophet · XGBoost · LSTM · Matplotlib · FastAPI**

---

## Overview

Modern computing systems continuously generate resource-utilization signals such as CPU, memory, disk, and network usage. Understanding how these resources are likely to behave in the near future can help identify capacity constraints, optimize infrastructure planning, and support proactive performance management.

This project develops an end-to-end time-series forecasting pipeline for system resource utilization.

Rather than relying on a single forecasting algorithm, the system compares statistical, machine-learning, and deep-learning approaches and provides a structured workflow for:

* Data preprocessing and validation
* Exploratory time-series analysis
* Temporal feature engineering
* Forecast model training
* Chronological model evaluation
* Model comparison and selection
* Forecast generation
* REST API-based inference

The project is structured as a modular machine-learning application rather than a notebook-only experiment.

---

## Problem Statement

System resources exhibit temporal patterns such as:

* Long-term trends
* Periodic behavior
* Short-term dependencies
* Seasonal fluctuations
* Sudden changes in utilization

A forecasting system should therefore be able to learn from historical observations while preserving the temporal ordering of the data.

The objective of this project is to build a forecasting pipeline capable of predicting future resource utilization while comparing different modeling approaches and exposing the resulting predictions through an API.

---

## Objectives

* Build a reusable time-series preprocessing pipeline.
* Analyze temporal trends and seasonality.
* Engineer lag, rolling, calendar, and trend-based features.
* Compare statistical, machine-learning, and deep-learning forecasting models.
* Evaluate models using time-aware validation.
* Automatically identify the strongest model according to the selected evaluation metric.
* Persist trained model artifacts for inference.
* Expose forecasting functionality through FastAPI.
* Provide visualizations for historical and predicted resource utilization.

---

## System Architecture

```text
                 Historical Resource Data
                          │
                          ▼
                ┌───────────────────┐
                │ Data Validation   │
                │ & Preprocessing   │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ Exploratory       │
                │ Time-Series       │
                │ Analysis          │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ Feature           │
                │ Engineering       │
                └─────────┬─────────┘
                          │
             ┌────────────┼────────────┐
             │            │            │
             ▼            ▼            ▼
          SARIMA       Prophet      XGBoost
             │            │            │
             └────────────┼────────────┘
                          │
                          ▼
                    ┌───────────┐
                    │   LSTM    │
                    └─────┬─────┘
                          │
                          ▼
                ┌───────────────────┐
                │ Model Evaluation  │
                │ & Comparison      │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ Best Model        │
                │ Selection         │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ Model Persistence │
                └─────────┬─────────┘
                          │
                          ▼
                    ┌──────────┐
                    │ FastAPI  │
                    │ REST API │
                    └────┬─────┘
                         │
                         ▼
                  Future Forecasts
```

---

## Forecasting Pipeline

The system follows a structured forecasting workflow:

### 1. Data Ingestion

Historical resource-utilization observations are loaded into the preprocessing pipeline.

### 2. Data Validation

The input data is checked for:

* Missing observations
* Invalid timestamps
* Duplicate records
* Missing target values
* Temporal continuity

### 3. Preprocessing

The preprocessing layer prepares the data for forecasting by handling missing values, timestamp ordering, and model-specific transformations.

### 4. Exploratory Analysis

Temporal analysis is used to identify:

* Trends
* Seasonality
* Distribution changes
* Lag relationships
* Resource utilization patterns

### 5. Feature Engineering

Time-aware features include:

* Lag features
* Rolling statistics
* Calendar features
* Trend indicators
* Seasonal indicators

Example lag features:

```text
lag_1
lag_7
lag_14
lag_30
```

Example rolling statistics:

```text
rolling_mean
rolling_std
```

### 6. Model Training

Multiple forecasting approaches are trained independently.

### 7. Evaluation

Models are evaluated using chronological validation rather than random train-test splitting.

### 8. Model Selection

The forecasting pipeline compares model performance and identifies the strongest candidate according to the selected metric.

### 9. Model Persistence

The selected model and required preprocessing artifacts are persisted for later inference.

### 10. API Inference

FastAPI provides a REST interface through which future resource-utilization forecasts can be requested.

---

## Models

### SARIMA

**Seasonal Autoregressive Integrated Moving Average**

Used as a classical statistical forecasting approach for modeling temporal dependencies, trend, and seasonality.

### Prophet

Used for trend- and seasonality-oriented forecasting with a model structure suited to business and operational time-series data.

### XGBoost

A gradient-boosting approach that converts the forecasting problem into a supervised-learning task using engineered temporal features.

### LSTM

A recurrent neural-network architecture designed to model sequential dependencies in time-series data.

---

## Model Comparison

| Model   | Modeling Approach | Primary Strength                    |
| ------- | ----------------- | ----------------------------------- |
| SARIMA  | Statistical       | Temporal dependency and seasonality |
| Prophet | Statistical       | Trend and seasonal structure        |
| XGBoost | Machine Learning  | Non-linear feature relationships    |
| LSTM    | Deep Learning     | Sequential dependencies             |

The final model is selected based on evaluation results rather than assuming that one forecasting technique will perform best for every dataset.

---

## Evaluation Strategy

Time-series data requires evaluation methods that preserve chronological ordering.

The project therefore avoids conventional random train-test splitting for forecasting evaluation.

Evaluation metrics include:

* **RMSE** — Root Mean Squared Error
* **MAE** — Mean Absolute Error
* **MAPE** — Mean Absolute Percentage Error

The evaluation pipeline is designed to support model comparison using a consistent validation strategy.

---

## Leakage Prevention

Preventing temporal leakage is a key part of the forecasting workflow.

The project follows safeguards such as:

* Chronological train-validation separation
* Past-only lag features
* Rolling statistics derived from historical observations
* Train-only fitting for model transformations
* Time-aware validation
* Recursive forecasting for multi-step prediction where applicable

These practices help ensure that future information is not unintentionally introduced during model training.

---

## FastAPI Service

The trained forecasting pipeline is exposed through a REST API using FastAPI.

Example service structure:

```text
FastAPI
   │
   ├── Health Check
   │
   ├── Forecast Request
   │
   ├── Input Validation
   │
   ├── Model Loading
   │
   └── Prediction Response
```

Example endpoints:

```text
GET /health
GET /api/v1/forecast/{resource}
```

Example request:

```text
GET /api/v1/forecast/cpu?horizon=8
```

Example response structure:

```json
{
  "resource": "cpu",
  "horizon": 8,
  "model": "selected_model",
  "predictions": []
}
```

> API response values will be populated after the forecasting pipeline is executed against the configured dataset.

---

## Project Structure

```text
system-resource-forecasting/
│
├── app/
│   ├── api/
│   │   └── routes.py
│   │
│   ├── core/
│   │   ├── config.py
│   │   └── logging.py
│   │
│   ├── data/
│   │   └── preprocessing.py
│   │
│   ├── features/
│   │   └── engineering.py
│   │
│   ├── models/
│   │   ├── sarima.py
│   │   ├── prophet.py
│   │   ├── xgboost.py
│   │   └── lstm.py
│   │
│   ├── schemas/
│   │   └── forecast.py
│   │
│   ├── services/
│   │   ├── training.py
│   │   ├── evaluation.py
│   │   ├── forecasting.py
│   │   └── model_registry.py
│   │
│   └── main.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── .gitkeep
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── outputs/
│   ├── figures/
│   └── reports/
│
├── scripts/
│   └── train.py
│
├── tests/
│   ├── test_api.py
│   ├── test_features.py
│   └── test_forecasting.py
│
├── docs/
│   └── architecture.md
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Tech Stack

### Programming

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* XGBoost

### Time-Series Forecasting

* SARIMA
* Prophet
* LSTM

### Backend

* FastAPI
* Pydantic
* REST APIs

### Visualization

* Matplotlib

### Engineering

* Git
* Modular Python architecture
* Automated model evaluation

---

## Example Workflow

```text
Historical Metrics
        │
        ▼
Data Preparation
        │
        ▼
Temporal Analysis
        │
        ▼
Feature Engineering
        │
        ▼
Model Training
        │
        ├── SARIMA
        ├── Prophet
        ├── XGBoost
        └── LSTM
        │
        ▼
Model Evaluation
        │
        ▼
Best Model Selection
        │
        ▼
Forecast Generation
        │
        ▼
FastAPI Inference
```

---

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the training pipeline:

```bash
python -m app.services.training
```

Start the API:

```bash
uvicorn app.main:app --reload
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

> Execution and environment-specific configuration are part of the subsequent validation phase of the project.

---

## Future Improvements

The project can be extended with:

* Prediction intervals
* Probabilistic forecasting
* Automated model retraining
* MLflow-based experiment tracking
* Model versioning
* Dockerized deployment
* Cloud deployment
* Forecast monitoring
* Data-drift detection
* Resource anomaly detection
* Capacity-planning dashboards

---

## Project Status

**Architecture & implementation foundation**

The repository is structured as an end-to-end forecasting application. Runtime validation, model benchmarking, deployment hardening, and production monitoring are part of the subsequent development cycle.

---

## Author

**Kanchukommala Naveen**

**GitHub:** https://github.com/naveen-nhce26
**LinkedIn:** https://www.linkedin.com/in/naveen-kanchukommala/
