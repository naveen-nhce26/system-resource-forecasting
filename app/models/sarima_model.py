"""
SARIMA (SARIMAX) model module.

Responsibilities:
- state-wise chronological train/validation split (no randomization)
- SARIMAX training per state
- multi-step forecasting
- evaluation (RMSE, MAE, MAPE)

Leakage prevention:
- all splits preserve chronology
- evaluation forecasts only future points relative to the training cutoff
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.statespace.sarimax import SARIMAXResults
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "Missing dependency `statsmodels`. Install it with: pip install -r requirements.txt"
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


def time_series_split(
    df: pd.DataFrame,
    *,
    val_size: int,
    date_col: str = "Date",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological split: last `val_size` rows into validation.
    """

    if val_size <= 0:
        raise ValueError("val_size must be > 0")
    df = df.sort_values(date_col)
    if len(df) <= val_size:
        raise ValueError("Not enough rows for the requested validation size.")
    return df.iloc[:-val_size].copy(), df.iloc[-val_size:].copy()


@dataclass(frozen=True)
class SarimaConfig:
    """
    Practical SARIMA configuration for SARIMAX.

    Notes:
    - Defaults are intentionally lightweight to prevent long optimizer runs.
    - Seasonal modeling is disabled by default and only enabled when explicitly
      requested AND the state has enough history.
    """

    # Lightweight default
    order: tuple[int, int, int] = (1, 1, 0)

    # Seasonality: disabled by default (0,0,0,0).
    # If enabled, typical weekly seasonality is period=52 for weekly data.
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0)
    allow_seasonal: bool = False
    seasonal_min_train_points: int = 2 * 52  # at least ~2 seasonal cycles for stability

    # Optimizer controls (prevents long loops)
    method: str = "lbfgs"
    maxiter: int = 50

    # Safety / stability
    enforce_stationarity: bool = False
    enforce_invertibility: bool = False
    min_train_points: int = 20


@dataclass(frozen=True)
class SarimaStateModel:
    """
    Trained SARIMAX model container for a single state.
    """

    state: str
    config: SarimaConfig
    fitted: SARIMAXResults


def _validate_input(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty.")
    for col in ("State", "Date", "Total"):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        raise ValueError("`Date` must be datetime. Run preprocessing first.")


def train_state_sarima(
    df_state: pd.DataFrame,
    *,
    config: SarimaConfig,
    target_col: str = "Total",
) -> SARIMAXResults:
    """
    Train SARIMAX model on a single state's time series.
    """

    series = (
        df_state.sort_values("Date")
        .set_index("Date")[target_col]
        .astype(float)
    )
    if series.isna().any():
        raise ValueError("Target contains NaNs. Ensure preprocessing filled missing totals.")
    if len(series) < config.min_train_points:
        raise ValueError(f"Not enough history for SARIMA. Need >= {config.min_train_points}, got {len(series)}.")

    seasonal_order = config.seasonal_order
    if config.allow_seasonal and seasonal_order[3] > 1:
        if len(series) < config.seasonal_min_train_points:
            logger.info(
                "SARIMA seasonal disabled (insufficient history). points=%s required=%s",
                len(series),
                config.seasonal_min_train_points,
            )
            seasonal_order = (0, 0, 0, 0)

    model = SARIMAX(
        series,
        order=config.order,
        seasonal_order=seasonal_order,
        trend="n",
        simple_differencing=True,
        enforce_stationarity=config.enforce_stationarity,
        enforce_invertibility=config.enforce_invertibility,
    )
    # Cap optimizer iterations to avoid very long fits.
    return model.fit(method=config.method, maxiter=config.maxiter, disp=False)


def forecast_state(
    fitted: SARIMAXResults,
    *,
    steps: int,
) -> np.ndarray:
    """
    Forecast `steps` ahead from the fitted model.
    """

    if steps <= 0:
        raise ValueError("steps must be > 0")
    fc = fitted.get_forecast(steps=steps)
    return np.asarray(fc.predicted_mean, dtype=float)


def train_and_evaluate_state(
    df_state: pd.DataFrame,
    *,
    config: SarimaConfig,
    val_size: int,
) -> tuple[SARIMAXResults, Dict[str, float]]:
    """
    Train on train split and evaluate on the validation horizon.
    """

    train_df, val_df = time_series_split(df_state, val_size=val_size, date_col="Date")
    t0 = time.time()
    logger.info("SARIMA start state=%s train_points=%s val_points=%s", str(train_df["State"].iloc[0]), len(train_df), len(val_df))
    fitted = train_state_sarima(train_df, config=config)
    preds = forecast_state(fitted, steps=len(val_df))
    y_true = val_df.sort_values("Date")["Total"].astype(float).to_numpy()

    metrics = {
        "rmse": rmse(y_true, preds),
        "mae": mae(y_true, preds),
        "mape": mape(y_true, preds),
    }
    elapsed = time.time() - t0
    logger.info("SARIMA end state=%s elapsed_s=%.2f", str(train_df["State"].iloc[0]), elapsed)
    return fitted, metrics


def train_models_by_state(
    df: pd.DataFrame,
    *,
    config: Optional[SarimaConfig] = None,
    val_size: Optional[int] = None,
) -> Dict[str, SarimaStateModel]:
    """
    Train SARIMA models for each state.
    """

    _validate_input(df)
    config = config or SarimaConfig()
    val_size = int(val_size or settings.forecast_horizon_weeks)

    models: Dict[str, SarimaStateModel] = {}
    for state, g in df.groupby("State", sort=False):
        try:
            logger.info("Training SARIMA for state=%s rows=%s", state, len(g))
            fitted, metrics = train_and_evaluate_state(g, config=config, val_size=val_size)
            logger.info("SARIMA metrics state=%s rmse=%.4f mae=%.4f mape=%.2f%%", state, metrics["rmse"], metrics["mae"], metrics["mape"])
            models[str(state)] = SarimaStateModel(state=str(state), config=config, fitted=fitted)
        except Exception:
            logger.exception("SARIMA training failed for state=%s", state)
            # Keep training robust: skip problematic states.
            continue
    return models


def predict_next_weeks_by_state(
    models: Dict[str, SarimaStateModel],
    *,
    steps: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Forecast the next `steps` points for each state.
    """

    steps = int(steps or settings.forecast_horizon_weeks)
    out: Dict[str, np.ndarray] = {}
    for state, m in models.items():
        try:
            out[state] = forecast_state(m.fitted, steps=steps)
        except Exception:
            logger.exception("SARIMA forecast failed for state=%s", state)
            raise
    return out

