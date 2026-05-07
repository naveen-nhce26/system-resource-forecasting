"""
LSTM model module for state-wise time-series forecasting.

Responsibilities:
- state-wise chronological train/validation split (no randomization)
- train-only scaling (prevents leakage)
- sequence generation for supervised learning
- LSTM training per state
- recursive multi-step forecasting
- evaluation (RMSE, MAE, MAPE)

Leakage prevention:
- chronology-preserving split only
- scaler is fit on train portion only
- validation/inference uses only past context for sequences
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
try:
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.optimizers import Adam
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "Missing dependency `tensorflow`. Install it with: pip install -r requirements.txt"
    ) from exc

from app.config import settings
from app.utils.logger import get_logger


logger = get_logger(__name__, level=settings.log_level, log_dir=settings.logs_dir)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, *, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def time_series_split(df: pd.DataFrame, *, val_size: int, date_col: str = "Date") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological split: last `val_size` rows into validation.
    """

    if val_size <= 0:
        raise ValueError("val_size must be > 0")
    df = df.sort_values(date_col)
    if len(df) <= val_size:
        raise ValueError("Not enough rows for the requested validation size.")
    return df.iloc[:-val_size].copy(), df.iloc[-val_size:].copy()


def _validate_input(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty.")
    for col in ("State", "Date", "Total"):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        raise ValueError("`Date` must be datetime. Run preprocessing first.")


def make_sequences(series: np.ndarray, *, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a 1D array into (X, y) sequences for next-step prediction.

    Shapes:
    - X: (n_samples, window, 1)
    - y: (n_samples,)
    """

    if window <= 0:
        raise ValueError("window must be > 0")
    if series.ndim != 1:
        raise ValueError("series must be 1D")
    if len(series) <= window:
        raise ValueError("Not enough data to build sequences with the given window.")

    X, y = [], []
    for i in range(window, len(series)):
        X.append(series[i - window : i])
        y.append(series[i])

    X_arr = np.asarray(X, dtype=float).reshape(-1, window, 1)
    y_arr = np.asarray(y, dtype=float)
    return X_arr, y_arr


@dataclass(frozen=True)
class LSTMConfig:
    """
    LSTM training configuration (lightweight, production-friendly defaults).
    """

    window: int = 30
    lstm_units: int = 64
    learning_rate: float = 1e-3
    epochs: int = 20
    batch_size: int = 32
    verbose: int = 0


@dataclass(frozen=True)
class LSTMStateModel:
    state: str
    config: LSTMConfig
    model: Sequential
    scaler: MinMaxScaler


def build_lstm_model(*, window: int, units: int, learning_rate: float) -> Sequential:
    """
    Build a simple LSTM for 1-step ahead forecasting.
    """

    m = Sequential(
        [
            LSTM(units, input_shape=(window, 1)),
            Dense(1),
        ]
    )
    m.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return m


def train_state(
    df_state: pd.DataFrame,
    *,
    config: LSTMConfig,
    val_size: int,
    target_col: str = "Total",
) -> tuple[Sequential, MinMaxScaler, Dict[str, float]]:
    """
    Train LSTM for a single state and evaluate on the validation horizon.
    """

    df_state = df_state.sort_values("Date")
    train_df, val_df = time_series_split(df_state, val_size=val_size, date_col="Date")

    y_train_raw = train_df[target_col].to_numpy(dtype=float)
    y_val_raw = val_df[target_col].to_numpy(dtype=float)
    if np.isnan(y_train_raw).any() or np.isnan(y_val_raw).any():
        raise ValueError("Target contains NaNs. Ensure preprocessing filled missing totals.")

    scaler = MinMaxScaler()
    y_train_scaled = scaler.fit_transform(y_train_raw.reshape(-1, 1)).reshape(-1)
    y_val_scaled = scaler.transform(y_val_raw.reshape(-1, 1)).reshape(-1)

    X_train, y_train = make_sequences(y_train_scaled, window=config.window)

    # Validation sequences need past context. Use tail from train + val.
    context = np.concatenate([y_train_scaled[-config.window :], y_val_scaled], axis=0)
    X_val, y_val = make_sequences(context, window=config.window)

    model = build_lstm_model(window=config.window, units=config.lstm_units, learning_rate=config.learning_rate)
    model.fit(
        X_train,
        y_train,
        epochs=config.epochs,
        batch_size=config.batch_size,
        verbose=config.verbose,
        shuffle=False,  # preserve order
    )

    val_pred_scaled = model.predict(X_val, verbose=0).reshape(-1)
    val_pred = scaler.inverse_transform(val_pred_scaled.reshape(-1, 1)).reshape(-1)
    y_val_true = scaler.inverse_transform(y_val.reshape(-1, 1)).reshape(-1)

    metrics = {
        "rmse": rmse(y_val_true, val_pred),
        "mae": mae(y_val_true, val_pred),
        "mape": mape(y_val_true, val_pred),
    }
    return model, scaler, metrics


def train_models_by_state(
    df: pd.DataFrame,
    *,
    config: Optional[LSTMConfig] = None,
    val_size: Optional[int] = None,
) -> Dict[str, LSTMStateModel]:
    """
    Train LSTM models per state on the univariate target series.
    """

    _validate_input(df)
    config = config or LSTMConfig()
    val_size = int(val_size or settings.forecast_horizon_weeks)

    models: Dict[str, LSTMStateModel] = {}
    for state, g in df.groupby("State", sort=False):
        try:
            logger.info("Training LSTM for state=%s rows=%s", state, len(g))
            model, scaler, metrics = train_state(g, config=config, val_size=val_size)
            logger.info("LSTM metrics state=%s rmse=%.4f mae=%.4f mape=%.2f%%", state, metrics["rmse"], metrics["mae"], metrics["mape"])
            models[str(state)] = LSTMStateModel(state=str(state), config=config, model=model, scaler=scaler)
        except Exception:
            logger.exception("LSTM training failed for state=%s", state)
            raise
    return models


def recursive_forecast_state(
    state_model: LSTMStateModel,
    df_state: pd.DataFrame,
    *,
    steps: int,
    target_col: str = "Total",
) -> np.ndarray:
    """
    Recursive multi-step forecast for a single state.

    Uses last `window` points as context, predicts next step, appends prediction,
    and repeats. This uses only past information at each step.
    """

    if steps <= 0:
        raise ValueError("steps must be > 0")

    df_state = df_state.sort_values("Date")
    y_raw = df_state[target_col].to_numpy(dtype=float)
    if np.isnan(y_raw).any():
        raise ValueError("Target contains NaNs. Ensure preprocessing filled missing totals.")

    scaler = state_model.scaler
    y_scaled = scaler.transform(y_raw.reshape(-1, 1)).reshape(-1)
    window = state_model.config.window
    if len(y_scaled) < window:
        raise ValueError("Not enough history to forecast with the configured window.")

    history = y_scaled.tolist()
    preds_scaled: list[float] = []
    for _ in range(steps):
        x = np.asarray(history[-window:], dtype=float).reshape(1, window, 1)
        yhat = float(state_model.model.predict(x, verbose=0).reshape(-1)[0])
        preds_scaled.append(yhat)
        history.append(yhat)

    preds = scaler.inverse_transform(np.asarray(preds_scaled).reshape(-1, 1)).reshape(-1)
    return preds.astype(float)


def predict_next_weeks_by_state(
    models: Dict[str, LSTMStateModel],
    df_history: pd.DataFrame,
    *,
    steps: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Forecast next `steps` points per state using recursive strategy.
    """

    _validate_input(df_history)
    steps = int(steps or settings.forecast_horizon_weeks)

    out: Dict[str, np.ndarray] = {}
    for state, m in models.items():
        try:
            df_state = df_history[df_history["State"].astype(str) == str(state)].sort_values("Date")
            out[state] = recursive_forecast_state(m, df_state, steps=steps)
        except Exception:
            logger.exception("LSTM forecast failed for state=%s", state)
            raise
    return out

