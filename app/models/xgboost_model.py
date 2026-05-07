"""
XGBoost model module for time-series forecasting using engineered features.

Expected input:
DataFrame with:
- identifiers: State, Date
- target: Total
- engineered features:
  lag_1, lag_7, lag_30, rolling_mean, rolling_std,
  month, quarter, year, holiday_flag, trend_index

Responsibilities:
- chronological train/validation split (no randomization)
- state-wise training and prediction
- multi-step forecasting via recursive strategy (predict next step, append, repeat)
- evaluation (RMSE, MAE, MAPE)

Leakage prevention:
- train/validation split preserves time order
- model uses only features available at prediction time
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
try:
    from xgboost import XGBRegressor
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "Missing dependency `xgboost`. Install it with: pip install -r requirements.txt"
    ) from exc

from app.config import settings
from app.utils.logger import get_logger


logger = get_logger(__name__, level=settings.log_level, log_dir=settings.logs_dir)


FEATURE_COLUMNS: tuple[str, ...] = (
    "lag_1",
    "lag_7",
    "lag_30",
    "rolling_mean",
    "rolling_std",
    "month",
    "quarter",
    "year",
    "holiday_flag",
    "trend_index",
)


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


def time_series_split(df: pd.DataFrame, *, val_size: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological split: last `val_size` rows into validation.
    """

    if val_size <= 0:
        raise ValueError("val_size must be > 0")
    df = df.sort_values("Date")
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
    missing_feats = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_feats:
        raise ValueError(f"Missing required feature columns: {missing_feats}")


@dataclass(frozen=True)
class XGBoostConfig:
    """
    XGBoost hyperparameters (sane defaults).
    """

    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 6
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    random_state: int = 42
    n_jobs: int = 0  # 0 lets xgboost decide / uses all cores on many setups


@dataclass(frozen=True)
class XGBoostStateModel:
    state: str
    config: XGBoostConfig
    model: XGBRegressor
    feature_columns: tuple[str, ...] = field(default=FEATURE_COLUMNS)


def _make_regressor(config: XGBoostConfig) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        random_state=config.random_state,
        n_jobs=config.n_jobs if config.n_jobs != 0 else None,
        objective="reg:squarederror",
    )


def train_state(
    df_state: pd.DataFrame,
    *,
    config: XGBoostConfig,
    val_size: int,
    target_col: str = "Total",
) -> tuple[XGBRegressor, Dict[str, float]]:
    """
    Train XGBoost on a single state's engineered features and evaluate on holdout.
    """

    train_df, val_df = time_series_split(df_state, val_size=val_size)

    X_train = train_df[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    y_train = train_df[target_col].to_numpy(dtype=float)
    X_val = val_df[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    y_val = val_df[target_col].to_numpy(dtype=float)

    model = _make_regressor(config)
    model.fit(X_train, y_train)
    preds = model.predict(X_val).astype(float)

    metrics = {
        "rmse": rmse(y_val, preds),
        "mae": mae(y_val, preds),
        "mape": mape(y_val, preds),
    }
    return model, metrics


def train_models_by_state(
    df_features: pd.DataFrame,
    *,
    config: Optional[XGBoostConfig] = None,
    val_size: Optional[int] = None,
) -> Dict[str, XGBoostStateModel]:
    """
    Train XGBoost models per state from a feature-engineered dataframe.
    """

    _validate_input(df_features)
    config = config or XGBoostConfig()
    val_size = int(val_size or settings.forecast_horizon_weeks)

    models: Dict[str, XGBoostStateModel] = {}
    for state, g in df_features.groupby("State", sort=False):
        try:
            g = g.sort_values("Date")
            logger.info("Training XGBoost for state=%s rows=%s", state, len(g))
            model, metrics = train_state(g, config=config, val_size=val_size)
            logger.info("XGBoost metrics state=%s rmse=%.4f mae=%.4f mape=%.2f%%", state, metrics["rmse"], metrics["mae"], metrics["mape"])
            models[str(state)] = XGBoostStateModel(state=str(state), config=config, model=model)
        except Exception:
            logger.exception("XGBoost training failed for state=%s", state)
            raise
    return models


def _next_date(last_date: pd.Timestamp, freq: str) -> pd.Timestamp:
    # pandas offset parsing handles common strings (e.g. "W", "D", "MS").
    # For weekly, this yields next week anchored to the same weekday.
    return pd.date_range(start=last_date, periods=2, freq=freq)[1]


def _infer_freq(dates: pd.Series) -> str:
    return pd.infer_freq(dates.sort_values().unique()) or "W"


def _holiday_flag_for_date(dt: pd.Timestamp) -> int:
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=dt.normalize(), end=dt.normalize())
    return int(dt.normalize() in(holidays))  # type: ignore[call-arg]


def recursive_forecast_state(
    state_model: XGBoostStateModel,
    df_state_features: pd.DataFrame,
    *,
    steps: int,
    target_col: str = "Total",
) -> np.ndarray:
    """
    Recursive multi-step forecast for a single state.

    Important:
    This function expects df_state_features to include the latest known row with
    populated feature columns. For subsequent steps, it updates lag/rolling/trend
    based on predicted values (not future ground truth), which avoids leakage.
    """

    if steps <= 0:
        raise ValueError("steps must be > 0")

    df = df_state_features.sort_values("Date").copy()
    freq = _infer_freq(df["Date"])

    required = set(state_model.feature_columns) | {"Date", "State", target_col}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for recursive forecast: {missing}")

    preds: List[float] = []

    # We'll build each future row's features from the running history.
    for _ in range(steps):
        last = df.iloc[-1]
        last_date = pd.Timestamp(last["Date"])
        next_date = pd.Timestamp(_next_date(last_date, freq))

        # Build features using available history (Total includes predicted values appended so far).
        hist_total = df[target_col].to_numpy(dtype=float)
        # Safe: all lags/rolling use only past history.
        lag_1 = hist_total[-1] if len(hist_total) >= 1 else np.nan
        lag_7 = hist_total[-7] if len(hist_total) >= 7 else np.nan
        lag_30 = hist_total[-30] if len(hist_total) >= 30 else np.nan

        # rolling computed on last 7 values excluding the future point; equivalent to past-only.
        window = hist_total[-7:] if len(hist_total) >= 7 else hist_total
        rolling_mean = float(np.mean(window)) if len(window) else np.nan
        rolling_std = float(np.std(window)) if len(window) else np.nan

        month = int(next_date.month)
        quarter = int(((next_date.month - 1) // 3) + 1)
        year = int(next_date.year)

        holiday_flag = _holiday_flag_for_date(next_date)

        trend_index = int(last["trend_index"]) + 1 if "trend_index" in df.columns else int(len(df))

        x_row = np.array(
            [[lag_1, lag_7, lag_30, rolling_mean, rolling_std, month, quarter, year, holiday_flag, trend_index]],
            dtype=float,
        )

        y_hat = float(state_model.model.predict(x_row)[0])
        preds.append(y_hat)

        # Append new row to history for subsequent steps.
        new_row = {
            "State": last["State"],
            "Date": next_date,
            target_col: y_hat,
            "lag_1": lag_1,
            "lag_7": lag_7,
            "lag_30": lag_30,
            "rolling_mean": rolling_mean,
            "rolling_std": rolling_std,
            "month": month,
            "quarter": quarter,
            "year": year,
            "holiday_flag": holiday_flag,
            "trend_index": trend_index,
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    return np.asarray(preds, dtype=float)


def predict_next_weeks_by_state(
    models: Dict[str, XGBoostStateModel],
    df_features: pd.DataFrame,
    *,
    steps: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Forecast next `steps` points per state using recursive strategy.
    """

    _validate_input(df_features)
    steps = int(steps or settings.forecast_horizon_weeks)

    out: Dict[str, np.ndarray] = {}
    for state, m in models.items():
        try:
            df_state = df_features[df_features["State"].astype(str) == str(state)].sort_values("Date")
            out[state] = recursive_forecast_state(m, df_state, steps=steps)
        except Exception:
            logger.exception("XGBoost forecast failed for state=%s", state)
            raise
    return out

