# State-wise Sales Forecasting Backend (FastAPI + ML)

## Project Objective

Forecast the next **8 weeks** of sales for each **state** using historical sales data.

The system:

* trains multiple forecasting models
* evaluates and compares model performance
* automatically selects the best-performing model
* saves the best model
* exposes predictions through a FastAPI REST API

The project is designed like a **production-oriented backend ML service** instead of a notebook-based workflow.

---

# Problem Statement

Given historical sales data grouped by `State` and `Date`, build a scalable forecasting backend capable of:

* handling missing dates and missing values
* performing time-series feature engineering
* training multiple forecasting algorithms
* evaluating models using time-series validation
* automatically selecting the best model
* serving predictions through an API

---

# Dataset

Input dataset:

```text
data/casestudy.xlsx
```

Main columns:

* `State`
* `Date`
* `Total` (target variable)
* `Category`

---

# Tech Stack

* Python
* FastAPI
* Pandas
* NumPy
* Statsmodels
* Prophet
* XGBoost
* TensorFlow / Keras
* Scikit-learn
* Matplotlib
* Joblib

---

# System Architecture

The application is divided into multiple layers:

* **Data Layer**

  * dataset loading
  * schema validation
  * missing value handling
  * state-wise continuity handling

* **Feature Engineering Layer**

  * lag features
  * rolling statistics
  * calendar features
  * holiday flags
  * trend features

* **Model Layer**

  * SARIMA
  * Prophet
  * XGBoost
  * LSTM

* **Pipeline Layer**

  * automated training
  * evaluation
  * leaderboard generation
  * best model selection
  * model persistence

* **Serving Layer**

  * forecasting service
  * FastAPI REST endpoints
  * prediction APIs

---

# Folder Structure

```text
datascience_casestudy/
│
├── app/
│   ├── api/
│   │   └── routes.py
│   │
│   ├── models/
│   │   ├── sarima_model.py
│   │   ├── prophet_model.py
│   │   ├── xgboost_model.py
│   │   └── lstm_model.py
│   │
│   ├── services/
│   │   ├── preprocessing.py
│   │   ├── feature_engineering.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── forecasting.py
│   │
│   ├── utils/
│   │   └── logger.py
│   │
│   ├── saved_models/
│   ├── logs/
│   ├── main.py
│   └── config.py
│
├── data/
│   └── casestudy.xlsx
│
├── outputs/
│   └── forecast.png
│
├── client.py
├── requirements.txt
└── README.md
```

---

# Models Implemented

## 1. SARIMA

Classical statistical forecasting model for seasonality and trend.

## 2. Facebook Prophet

Business-oriented forecasting model for trend and seasonality handling.

## 3. XGBoost

Machine learning forecasting model using engineered lag and rolling features.

## 4. LSTM

Deep learning sequential forecasting model.

All models are:

* trained state-wise
* evaluated using chronological validation
* compared automatically

---

# Feature Engineering

The following features are generated state-wise:

## Lag Features

* `lag_1`
* `lag_7`
* `lag_30`

## Rolling Statistics

* rolling mean
* rolling standard deviation

## Calendar Features

* month
* quarter
* year
* day of week

## Holiday Features

* holiday flag using US federal holidays

## Trend Features

* trend index

---

# Time-Series Leakage Prevention

The system avoids data leakage using proper time-series validation logic.

Implemented safeguards:

* no random train-test split
* validation uses last chronological observations
* lag features use only past values
* rolling statistics exclude current observations
* LSTM scaling uses train-only fit
* recursive multi-step forecasting for future predictions

---

# Automated Training Pipeline

Training entry point:

```bash
python -m app.services.train --horizon 8 --metric rmse
```

Pipeline steps:

1. load dataset
2. preprocess data
3. feature engineering
4. train SARIMA
5. train Prophet
6. train XGBoost
7. train LSTM
8. evaluate models
9. generate leaderboard
10. select best model
11. save best model artifact

---

# Evaluation Metrics

Models are evaluated using:

* RMSE
* MAE
* MAPE

Example leaderboard:

| Model   | RMSE |
| ------- | ---- |
| SARIMA  | XXX  |
| Prophet | XXX  |
| XGBoost | XXX  |
| LSTM    | XXX  |

Best model is automatically selected based on evaluation score.

---

# FastAPI Backend

Run API:

```bash
uvicorn app.main:app --reload
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

Available endpoints:

* `GET /health`
* `GET /api/v1/forecast/{state}`

---

# Forecast API Example

Request:

```text
GET /api/v1/forecast/Texas?weeks=8
```

Response:

```json
{
  "state": "Texas",
  "forecast_weeks": 8,
  "best_model": "XGBoost",
  "predictions": [
    919442624,
    899336000,
    902312448
  ]
}
```

---

# Terminal Client Application

A lightweight terminal client (`client.py`) is included for demonstration purposes.

The client:

* accepts user input
* calls the FastAPI forecasting API
* displays predictions in terminal
* generates forecast visualization graph

Run:

```bash
python client.py
```

---

# Sample Terminal Output

```text
==============================
 SALES FORECASTING SYSTEM
==============================

Enter State Name: Texas
Enter Forecast Weeks: 8

========== FORECAST RESULT ==========

State           : Texas
Forecast Weeks  : 8
Best Model      : XGBoost

Predictions:

Week 1: 919,442,624.00
Week 2: 899,336,000.00
Week 3: 902,312,448.00
...
```

---

# Forecast Visualization

The system generates:

```text
forecast.png
```

showing future sales trend predictions.

---

# Setup Instructions

Install dependencies:

```bash
pip install -r requirements.txt
```

Run training:

```bash
python -m app.services.train --horizon 8 --metric rmse
```

Run API:

```bash
uvicorn app.main:app --reload
```

Run client application:

```bash
python client.py
```

---

# Future Improvements

* prediction intervals
* model versioning
* scheduled retraining
* leaderboard persistence
* Docker deployment
* cloud deployment
* monitoring dashboards

---

# Conclusion

This project demonstrates:

* end-to-end time series forecasting
* production-oriented ML engineering
* automated model benchmarking
* feature engineering
* REST API integration
* scalable backend architecture
* forecasting visualization

The system was designed to simulate a real-world backend forecasting service rather than a notebook-based academic workflow.
