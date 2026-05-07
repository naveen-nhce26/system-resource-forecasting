"""
Evaluation utilities for time-series forecasting models.

This module provides:
- metrics (RMSE, MAE, MAPE)
- state-wise evaluation helpers
- model leaderboard generation and best-model selection

Leakage prevention:
- evaluation always uses a chronological holdout (last `val_size` rows) and
  predictions only target that holdout horizon.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

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


def time_series_split(df: pd.DataFrame, *, val_size: int, date_col: str = "Date") -> tuple[pd.DataFrame, pd.DataFrame]:
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
class ModelMetrics:
    """
    Aggregated metrics for a model.
    """

    rmse: float
    mae: float
    mape: float
    n_states: int


def _aggregate_state_metrics(per_state: Dict[str, Dict[str, float]]) -> ModelMetrics:
    rmses = [m["rmse"] for m in per_state.values()]
    maes = [m["mae"] for m in per_state.values()]
    mapes = [m["mape"] for m in per_state.values()]
    return ModelMetrics(
        rmse=float(np.mean(rmses)) if rmses else float("nan"),
        mae=float(np.mean(maes)) if maes else float("nan"),
        mape=float(np.mean(mapes)) if mapes else float("nan"),
        n_states=len(per_state),
    )


def leaderboard(
    model_metrics: Dict[str, ModelMetrics],
    *,
    primary_metric: str = "rmse",
) -> pd.DataFrame:
    """
    Create a leaderboard table sorted by `primary_metric`.
    """

    if primary_metric not in {"rmse", "mae", "mape"}:
        raise ValueError("primary_metric must be one of: rmse, mae, mape")

    rows = []
    for name, m in model_metrics.items():
        rows.append(
            {
                "Model": name,
                "RMSE": m.rmse,
                "MAE": m.mae,
                "MAPE": m.mape,
                "States": m.n_states,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    sort_col = {"rmse": "RMSE", "mae": "MAE", "mape": "MAPE"}[primary_metric]
    return df.sort_values(sort_col, ascending=True).reset_index(drop=True)


def select_best_model(model_metrics: Dict[str, ModelMetrics], *, primary_metric: str = "rmse") -> str:
    """
    Select best model name by minimum `primary_metric`.
    """

    if not model_metrics:
        raise ValueError("No model metrics provided.")
    if primary_metric not in {"rmse", "mae", "mape"}:
        raise ValueError("primary_metric must be one of: rmse, mae, mape")

    def _key(item: tuple[str, ModelMetrics]) -> float:
        name, m = item
        return float(getattr(m, primary_metric))

    best = min(model_metrics.items(), key=_key)[0]
    return best

