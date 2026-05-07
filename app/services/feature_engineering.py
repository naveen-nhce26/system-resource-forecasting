"""
Feature engineering for time-series forecasting (Phase 2).

Mandatory features (state-wise, leakage-safe):
- lag_1, lag_7, lag_30
- rolling_mean, rolling_std (past-only)
- month, quarter, year
- holiday_flag (US federal holidays via pandas calendar)
- trend_index (monotonic per state)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from app.config import settings
from app.utils.logger import get_logger


logger = get_logger(__name__, level=settings.log_level, log_dir=settings.logs_dir)


@dataclass(frozen=True)
class FeatureEngineeringResult:
    """
    Container for feature engineering outputs (extensible).
    """

    input: pd.DataFrame
    features: pd.DataFrame


def _validate_inputs(df: pd.DataFrame) -> None:
    """
    Validate required columns for feature generation.
    """

    if df is None or df.empty:
        raise ValueError("Input dataframe is empty.")
    for col in ("State", "Date", "Total"):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        raise ValueError("`Date` must be a datetime dtype. Run preprocessing first.")


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar features derived from `Date`.
    """

    out = df.copy()
    out["month"] = out["Date"].dt.month.astype("int16")
    out["quarter"] = out["Date"].dt.quarter.astype("int8")
    out["year"] = out["Date"].dt.year.astype("int16")
    return out


def add_holiday_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add `holiday_flag` using US federal holidays calendar.
    """

    out = df.copy()
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=out["Date"].min(), end=out["Date"].max())
    out["holiday_flag"] = out["Date"].dt.normalize().isin(holidays).astype("int8")
    return out


def add_trend_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a monotonic `trend_index` per state based on chronological ordering.
    """

    out = df.sort_values(["State", "Date"]).copy()
    out["trend_index"] = out.groupby("State").cumcount().astype("int32")
    return out


def add_lag_features(df: pd.DataFrame, *, target_col: str = "Total") -> pd.DataFrame:
    """
    Add mandatory lag features (state-wise).

    Leakage prevention:
    - Lags are computed with `shift(k)` within each state, so each row uses only
      strictly earlier observations.
    """

    out = df.sort_values(["State", "Date"]).copy()
    g = out.groupby("State")[target_col]
    out["lag_1"] = g.shift(1)
    out["lag_7"] = g.shift(7)
    out["lag_30"] = g.shift(30)
    return out


def add_rolling_features(
    df: pd.DataFrame,
    *,
    target_col: str = "Total",
    window: int = 7,
    min_periods: int = 1,
) -> pd.DataFrame:
    """
    Add rolling statistics features (state-wise).

    Leakage prevention:
    - Rolling statistics are computed on `shift(1)` so the current observation
      is excluded. This ensures only past data contributes to the feature.
    """

    out = df.sort_values(["State", "Date"]).copy()
    shifted = out.groupby("State")[target_col].shift(1)
    rolled = shifted.groupby(out["State"]).rolling(window=window, min_periods=min_periods)
    out["rolling_mean"] = rolled.mean().reset_index(level=0, drop=True)
    out["rolling_std"] = rolled.std(ddof=0).reset_index(level=0, drop=True)
    return out


def finalize_feature_matrix(
    df: pd.DataFrame,
    *,
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Finalize feature matrix by handling NaNs introduced by lagging/rolling.

    Args:
        drop_na: If True, drop rows where any mandatory feature is NaN.
            This is appropriate for training. For inference, set False and
            handle NaNs upstream based on available history.
    """

    mandatory = ["lag_1", "lag_7", "lag_30", "rolling_mean", "rolling_std"]
    out = df.copy()
    if drop_na:
        before = len(out)
        out = out.dropna(subset=mandatory).reset_index(drop=True)
        logger.info("Dropped %s rows due to NaNs from lags/rolling.", before - len(out))
    return out


def build_features(df: pd.DataFrame) -> FeatureEngineeringResult:
    """
    Feature engineering entrypoint.

    Generates state-wise, leakage-safe time-series features.
    """

    try:
        _validate_inputs(df)
        out = df.copy()
        out = add_calendar_features(out)
        out = add_holiday_flag(out)
        out = add_trend_index(out)
        out = add_lag_features(out, target_col="Total")
        out = add_rolling_features(out, target_col="Total", window=7, min_periods=1)
        out = finalize_feature_matrix(out, drop_na=True)

        logger.info(
            "Feature engineering complete. input_shape=%s feature_shape=%s",
            df.shape,
            out.shape,
        )
        return FeatureEngineeringResult(input=df, features=out)
    except Exception:
        logger.exception("Feature engineering failed.")
        raise

