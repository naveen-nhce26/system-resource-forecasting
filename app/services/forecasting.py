"""
Forecasting service.

Provides:
- best model artifact loading
- state-wise next-N prediction (default: next 8 weeks)
- prediction formatting for downstream API usage

This module intentionally focuses on *serving* forecasts using saved models.
Training/selection is handled by `app/services/train.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

from app.config import settings
from app.services.preprocessing import load_dataset, run_preprocessing
from app.services.feature_engineering import build_features
from app.utils.logger import get_logger


logger = get_logger(__name__, level=settings.log_level, log_dir=settings.logs_dir)


@dataclass(frozen=True)
class ForecastResult:
    """
    Standard forecast output format.
    """

    model_name: str
    horizon: int
    forecasts_by_state: Dict[str, Any]


def _saved_models_dir() -> Path:
    settings.saved_models_dir.mkdir(parents=True, exist_ok=True)
    return settings.saved_models_dir


def load_best_model_artifact(saved_dir: Optional[Path] = None) -> dict:
    """
    Load the persisted "best model" artifact.

    Returns a dict with:
    - model_name
    - models_by_state (for non-LSTM)
    OR for LSTM:
    - model_name="LSTM"
    - lstm_dir with per-state .keras files + meta joblib
    """

    saved_dir = Path(saved_dir) if saved_dir is not None else _saved_models_dir()

    # Non-LSTM path
    joblib_path = saved_dir / "best_model.joblib"
    if joblib_path.exists():
        return joblib.load(joblib_path)

    # LSTM path
    lstm_dir = saved_dir / "best_lstm"
    meta_path = lstm_dir / "lstm_meta.joblib"
    if meta_path.exists():
        meta = joblib.load(meta_path)
        meta["lstm_dir"] = lstm_dir
        return meta

    raise FileNotFoundError("No saved best model found in app/saved_models/. Run training pipeline first.")


def _load_lstm_models(meta: dict) -> Dict[str, LSTMStateModel]:
    """
    Load per-state LSTM models + scalers from the LSTM artifact directory.
    """

    from app.models.lstm_model import LSTMStateModel

    try:
        from tensorflow.keras.models import load_model
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency `tensorflow`. Install it with: pip install -r requirements.txt"
        ) from exc

    lstm_dir = Path(meta["lstm_dir"])
    states = meta.get("states", [])
    out: Dict[str, LSTMStateModel] = {}
    for state in states:
        model_path = lstm_dir / f"{state}.keras"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing LSTM model file for state={state}: {model_path}")
        cfg = meta[state]["config"]
        scaler = meta[state]["scaler"]
        model = load_model(model_path)
        out[str(state)] = LSTMStateModel(state=str(state), config=cfg, model=model, scaler=scaler)
    return out


def forecast_for_state(
    *,
    state: str,
    steps: Optional[int] = None,
    dataset_path: Optional[Path] = None,
) -> tuple[str, np.ndarray]:
    """
    Forecast next `steps` points for a single state using the saved best model.

    Returns:
        (best_model_name, predictions_array)
    """

    steps = int(steps or settings.forecast_horizon_weeks)
    raw = load_dataset(dataset_path)
    pre = run_preprocessing(raw).processed

    artifact = load_best_model_artifact()
    model_name = artifact["model_name"]

    if model_name in {"SARIMA", "Prophet", "XGBoost"}:
        from app.models.prophet_model import predict_next_weeks_by_state as prophet_predict
        from app.models.sarima_model import predict_next_weeks_by_state as sarima_predict
        from app.models.xgboost_model import predict_next_weeks_by_state as xgb_predict

        models_by_state = artifact["models_by_state"]
        if state not in models_by_state:
            raise KeyError(f"State not found in trained models: {state}")

        if model_name == "SARIMA":
            preds = sarima_predict({state: models_by_state[state]}, steps=steps)[state]
            return model_name, np.asarray(preds, dtype=float)

        if model_name == "Prophet":
            df_state = pre[pre["State"].astype(str) == str(state)].copy()
            fc = prophet_predict({state: models_by_state[state]}, df_state, steps=steps)[state]
            return model_name, fc["yhat"].to_numpy(dtype=float)

        feats = build_features(pre).features
        df_state_feats = feats[feats["State"].astype(str) == str(state)].copy()
        preds = xgb_predict({state: models_by_state[state]}, df_state_feats, steps=steps)[state]
        return model_name, np.asarray(preds, dtype=float)

    if model_name == "LSTM":
        from app.models.lstm_model import recursive_forecast_state as lstm_forecast_state

        models_by_state = _load_lstm_models(artifact)
        if state not in models_by_state:
            raise KeyError(f"State not found in trained models: {state}")

        df_state = pre[pre["State"].astype(str) == str(state)].sort_values("Date")
        preds = lstm_forecast_state(models_by_state[state], df_state, steps=steps)
        return "LSTM", np.asarray(preds, dtype=float)

    raise ValueError(f"Unsupported saved model_name={model_name!r}")


def forecast_next_weeks(
    *,
    steps: Optional[int] = None,
    dataset_path: Optional[Path] = None,
) -> ForecastResult:
    """
    Load data, prepare required inputs, load best model artifact, and forecast.
    """

    steps = int(steps or settings.forecast_horizon_weeks)
    raw = load_dataset(dataset_path)
    pre = run_preprocessing(raw).processed

    artifact = load_best_model_artifact()
    model_name = artifact["model_name"]

    if model_name in {"SARIMA", "Prophet", "XGBoost"}:
        from app.models.prophet_model import predict_next_weeks_by_state as prophet_predict
        from app.models.sarima_model import predict_next_weeks_by_state as sarima_predict
        from app.models.xgboost_model import predict_next_weeks_by_state as xgb_predict

        models_by_state = artifact["models_by_state"]

        if model_name == "SARIMA":
            forecasts = sarima_predict(models_by_state, steps=steps)
            return ForecastResult(model_name=model_name, horizon=steps, forecasts_by_state=forecasts)

        if model_name == "Prophet":
            forecasts = prophet_predict(models_by_state, pre, steps=steps)
            return ForecastResult(model_name=model_name, horizon=steps, forecasts_by_state=forecasts)

        # XGBoost needs engineered features
        feats = build_features(pre).features
        forecasts = xgb_predict(models_by_state, feats, steps=steps)
        return ForecastResult(model_name=model_name, horizon=steps, forecasts_by_state=forecasts)

    if model_name == "LSTM":
        from app.models.lstm_model import recursive_forecast_state as lstm_forecast_state

        models_by_state = _load_lstm_models(artifact)
        out: Dict[str, np.ndarray] = {}
        for state, sm in models_by_state.items():
            df_state = pre[pre["State"].astype(str) == str(state)].sort_values("Date")
            out[state] = lstm_forecast_state(sm, df_state, steps=steps)
        return ForecastResult(model_name="LSTM", horizon=steps, forecasts_by_state=out)

    raise ValueError(f"Unsupported saved model_name={model_name!r}")


def format_forecasts_long(
    forecasts: ForecastResult,
    *,
    last_observed_date_by_state: Dict[str, pd.Timestamp],
    freq: str = "W",
) -> pd.DataFrame:
    """
    Convert forecasts into a long dataframe: State, Date, yhat.

    Notes:
    - `last_observed_date_by_state` must be computed from the same dataset used
      to generate forecasts.
    - For Prophet, the internal result already contains dates; callers can skip
      this and format Prophet outputs directly if desired.
    """

    rows = []
    for state, yhat in forecasts.forecasts_by_state.items():
        if isinstance(yhat, pd.DataFrame) and "ds" in yhat.columns and "yhat" in yhat.columns:
            # Prophet output
            for _, r in yhat.iterrows():
                rows.append({"State": state, "Date": pd.Timestamp(r["ds"]), "yhat": float(r["yhat"]), "Model": forecasts.model_name})
            continue

        series = np.asarray(yhat, dtype=float).reshape(-1)
        start = pd.Timestamp(last_observed_date_by_state[state])
        future_dates = pd.date_range(start=start, periods=len(series) + 1, freq=freq)[1:]
        for d, v in zip(future_dates, series):
            rows.append({"State": state, "Date": pd.Timestamp(d), "yhat": float(v), "Model": forecasts.model_name})

    return pd.DataFrame(rows).sort_values(["State", "Date"]).reset_index(drop=True)

