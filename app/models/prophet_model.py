"""
Prophet model module.

Responsibilities:
- state-wise chronological train/validation split (no randomization)
- Prophet training per state
- multi-step forecasting
- evaluation (RMSE, MAE, MAPE)

Leakage prevention:
- chronological split only
- evaluation predicts only future dates relative to training cutoff
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
try:
    from prophet import Prophet
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "Missing dependency `prophet`. Install it with: pip install -r requirements.txt"
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


def infer_freq(dates: pd.Series) -> str:
    inferred = pd.infer_freq(dates.sort_values().unique())
    return inferred or "W"


@dataclass(frozen=True)
class ProphetConfig:
    """
    Prophet configuration (common knobs only).
    """

    yearly_seasonality: str | bool | int = "auto"
    weekly_seasonality: str | bool | int = "auto"
    daily_seasonality: str | bool | int = "auto"
    seasonality_mode: str = "additive"  # or "multiplicative"
    changepoint_prior_scale: float = 0.05


@dataclass(frozen=True)
class ProphetStateModel:
    state: str
    config: ProphetConfig
    model: Prophet
    freq: str


def _validate_input(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty.")
    for col in ("State", "Date", "Total"):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        raise ValueError("`Date` must be datetime. Run preprocessing first.")


def to_prophet_frame(df_state: pd.DataFrame, *, date_col: str = "Date", target_col: str = "Total") -> pd.DataFrame:
    """
    Convert to Prophet's required column names: ds, y.
    """

    out = df_state.sort_values(date_col)[[date_col, target_col]].copy()
    out = out.rename(columns={date_col: "ds", target_col: "y"})
    out["y"] = pd.to_numeric(out["y"], errors="coerce").astype(float)
    if out["y"].isna().any():
        raise ValueError("Target contains NaNs. Ensure preprocessing filled missing totals.")
    return out


def train_state_prophet(df_state: pd.DataFrame, *, config: ProphetConfig) -> tuple[Prophet, str]:
    """
    Train Prophet model for a single state.
    """

    hist = to_prophet_frame(df_state)
    freq = infer_freq(hist["ds"])

    m = Prophet(
        yearly_seasonality=config.yearly_seasonality,
        weekly_seasonality=config.weekly_seasonality,
        daily_seasonality=config.daily_seasonality,
        seasonality_mode=config.seasonality_mode,
        changepoint_prior_scale=config.changepoint_prior_scale,
    )
    m.fit(hist)
    return m, freq


def forecast_state(model: Prophet, *, last_date: pd.Timestamp, steps: int, freq: str) -> pd.DataFrame:
    """
    Forecast the next `steps` points using Prophet.

    Returns a dataframe with ds, yhat, yhat_lower, yhat_upper for the forecast horizon.
    """

    if steps <= 0:
        raise ValueError("steps must be > 0")

    future = model.make_future_dataframe(periods=steps, freq=freq, include_history=False)
    fc = model.predict(future)
    return fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()


def train_and_evaluate_state(
    df_state: pd.DataFrame,
    *,
    config: ProphetConfig,
    val_size: int,
) -> tuple[Prophet, str, Dict[str, float]]:
    """
    Train on train split and evaluate on validation dates.
    """

    train_df, val_df = time_series_split(df_state, val_size=val_size, date_col="Date")
    model, freq = train_state_prophet(train_df, config=config)

    val_frame = to_prophet_frame(val_df)
    # Predict exactly on validation timestamps to evaluate.
    preds = model.predict(val_frame[["ds"]])["yhat"].to_numpy(dtype=float)
    y_true = val_frame["y"].to_numpy(dtype=float)

    metrics = {
        "rmse": rmse(y_true, preds),
        "mae": mae(y_true, preds),
        "mape": mape(y_true, preds),
    }
    return model, freq, metrics


def train_models_by_state(
    df: pd.DataFrame,
    *,
    config: Optional[ProphetConfig] = None,
    val_size: Optional[int] = None,
) -> Dict[str, ProphetStateModel]:
    """
    Train Prophet models for each state.
    """

    _validate_input(df)
    config = config or ProphetConfig()
    val_size = int(val_size or settings.forecast_horizon_weeks)

    models: Dict[str, ProphetStateModel] = {}
    for state, g in df.groupby("State", sort=False):
        try:
            logger.info("Training Prophet for state=%s rows=%s", state, len(g))
            model, freq, metrics = train_and_evaluate_state(g, config=config, val_size=val_size)
            logger.info("Prophet metrics state=%s rmse=%.4f mae=%.4f mape=%.2f%%", state, metrics["rmse"], metrics["mae"], metrics["mape"])
            models[str(state)] = ProphetStateModel(state=str(state), config=config, model=model, freq=freq)
        except Exception:
            logger.exception("Prophet training failed for state=%s", state)
            raise
    return models


def predict_next_weeks_by_state(
    models: Dict[str, ProphetStateModel],
    df_history: pd.DataFrame,
    *,
    steps: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Forecast the next `steps` points per state.

    df_history is used only to locate each state's last observed date.
    """

    _validate_input(df_history)
    steps = int(steps or settings.forecast_horizon_weeks)

    out: Dict[str, pd.DataFrame] = {}
    for state, m in models.items():
        try:
            last_date = (
                df_history[df_history["State"].astype(str) == str(state)]
                .sort_values("Date")["Date"]
                .iloc[-1]
            )
            out[state] = forecast_state(m.model, last_date=last_date, steps=steps, freq=m.freq)
        except Exception:
            logger.exception("Prophet forecast failed for state=%s", state)
            raise
    return out

