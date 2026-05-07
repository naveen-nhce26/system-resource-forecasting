"""
Dataset loading + validation + preprocessing pipeline (Phase 2).

Production-grade preprocessing for time-series forecasting:
- safe Excel loading
- schema validation
- datetime parsing
- state-wise chronological ordering
- state-wise continuity (missing dates)
- missing value handling without introducing randomness

Important leakage note:
This module does not perform train/test splitting. Any imputation/interpolation
that can use future information is gated behind an optional `as_of` cutoff to
allow "past-only" filling for training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from app.config import settings
from app.utils.logger import get_logger


logger = get_logger(__name__, level=settings.log_level, log_dir=settings.logs_dir)


class DatasetValidationError(ValueError):
    """
    Raised when the input dataset fails schema/quality checks.
    """


@dataclass(frozen=True)
class PreprocessingResult:
    """
    Container for preprocessing outputs (extensible).
    """

    raw: pd.DataFrame
    processed: pd.DataFrame


REQUIRED_COLUMNS: tuple[str, ...] = ("State", "Date", "Total", "Category")


def load_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the historical sales dataset.

    Args:
        path: Optional explicit file path. Defaults to `settings.dataset_path`.

    Returns:
        DataFrame containing the raw dataset.
    """

    dataset_path = Path(path) if path is not None else settings.dataset_path
    logger.info("Loading dataset from %s", dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    try:
        # File format expected: Excel (as per `casestudy.xlsx`).
        return pd.read_excel(dataset_path, engine="openpyxl")
    except Exception as exc:
        logger.exception("Failed to read Excel file: %s", dataset_path)
        raise


def validate_required_columns(df: pd.DataFrame, required: Iterable[str] = REQUIRED_COLUMNS) -> None:
    """
    Validate that the dataset contains all required columns.
    """

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DatasetValidationError(f"Missing required columns: {missing}")


def validate_dataset(df: pd.DataFrame) -> None:
    """
    Validate dataset shape and required columns.
    """

    if df is None or df.empty:
        raise DatasetValidationError("Dataset is empty.")

    validate_required_columns(df)
    logger.info("Dataset loaded with shape=%s", df.shape)


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce core columns to stable dtypes:
    - `Date` -> datetime64[ns] (UTC-naive)
    - `State`, `Category` -> string
    - `Total` -> float
    """

    out = df.copy()
    out["State"] = out["State"].astype("string")
    out["Category"] = out["Category"].astype("string")

    # Robust datetime parsing; invalid parses become NaT and are rejected.
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    if out["Date"].isna().any():
        bad_rows = int(out["Date"].isna().sum())
        raise DatasetValidationError(f"Found {bad_rows} rows with invalid Date values.")

    out["Total"] = pd.to_numeric(out["Total"], errors="coerce").astype(float)
    return out


def sort_chronologically(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort the dataset state-wise by time.
    """

    return df.sort_values(["State", "Date"], ascending=[True, True]).reset_index(drop=True)


def infer_state_frequency(dates: pd.Series) -> str:
    """
    Infer a reasonable frequency string from a date series.

    Falls back to weekly ("W") when inference fails.
    """

    inferred = pd.infer_freq(dates.sort_values().unique())
    if inferred:
        return inferred
    return "W"


def ensure_state_continuity(
    df: pd.DataFrame,
    *,
    freq: Optional[str] = None,
    fill_category: str = "ffill",
) -> pd.DataFrame:
    """
    Ensure each state's time series has a continuous Date index by reindexing to
    a complete date range per state.

    Notes:
    - This does not aggregate duplicates. If your dataset has multiple rows per
      (State, Date), deduplicate/aggregate upstream before calling this.
    - `Category` is forward-filled by default within each state.
    """

    if df.duplicated(subset=["State", "Date"]).any():
        dup_count = int(df.duplicated(subset=["State", "Date"]).sum())
        raise DatasetValidationError(
            f"Found {dup_count} duplicate (State, Date) rows. Aggregate/deduplicate before continuity enforcement."
        )

    pieces: list[pd.DataFrame] = []
    for state, g in df.groupby("State", sort=False):
        g = g.sort_values("Date")
        series_freq = freq or infer_state_frequency(g["Date"])
        full_index = pd.date_range(start=g["Date"].min(), end=g["Date"].max(), freq=series_freq)
        g2 = g.set_index("Date").reindex(full_index)
        g2.index.name = "Date"
        g2["State"] = state

        if "Category" in g2.columns:
            if fill_category == "ffill":
                g2["Category"] = g2["Category"].ffill()
            elif fill_category == "bfill":
                g2["Category"] = g2["Category"].bfill()
            elif fill_category == "none":
                pass
            else:
                raise ValueError(f"Unsupported fill_category={fill_category!r}")

        pieces.append(g2.reset_index())

    out = pd.concat(pieces, ignore_index=True)
    out = sort_chronologically(out)
    return out


def fill_missing_totals(
    df: pd.DataFrame,
    *,
    as_of: Optional[pd.Timestamp] = None,
    strategy: str = "ffill_then_interpolate_inside",
) -> pd.DataFrame:
    """
    Fill missing `Total` values state-wise.

    Args:
        df: Preprocessed dataframe with required columns and continuity enforced.
        as_of: Optional cutoff timestamp for leakage-safe filling. If provided,
            filling/interpolation is performed using data up to `as_of` only.
            Values after `as_of` are left untouched.
        strategy:
            - "ffill": forward-fill only (past-only, leakage-safe)
            - "ffill_then_interpolate_inside": forward-fill, then time interpolation
              only for gaps that remain inside the observed history window.

    Leakage prevention:
    - Forward-fill uses only past values.
    - Interpolation is restricted to inside-gaps and can be limited to `as_of`
      when training so that future observations do not influence earlier rows.
    """

    out = df.copy()

    if as_of is not None:
        as_of = pd.Timestamp(as_of)

    def _fill_one(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("Date").copy()
        mask = slice(None)
        if as_of is not None:
            mask = g["Date"] <= as_of

        # Forward-fill is always safe (uses only past).
        g.loc[mask, "Total"] = g.loc[mask, "Total"].ffill()

        if strategy == "ffill":
            return g

        if strategy != "ffill_then_interpolate_inside":
            raise ValueError(f"Unsupported strategy={strategy!r}")

        # Interpolate only inside gaps (between known points) within the mask.
        # This avoids filling leading/trailing NaNs and reduces the risk of
        # introducing bias at the boundaries.
        s = g.loc[mask].set_index("Date")["Total"]
        g.loc[mask, "Total"] = (
            s.interpolate(method="time", limit_area="inside")
            .ffill()
            .values
        )
        return g

    out = out.groupby("State", group_keys=False, sort=False).apply(_fill_one)
    return out.reset_index(drop=True)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final missing value handling for non-target columns.
    """

    out = df.copy()
    # State is enforced. Category may remain missing at the start of a series.
    out["Category"] = out["Category"].fillna("unknown").astype("string")
    return out


def run_preprocessing(df: pd.DataFrame) -> PreprocessingResult:
    """
    Preprocessing pipeline entrypoint.

    Produces a cleaned, chronologically ordered, state-continuous dataset with
    stable types and missing value handling.
    """

    try:
        validate_dataset(df)
        typed = coerce_types(df)
        sorted_df = sort_chronologically(typed)
        continuous = ensure_state_continuity(sorted_df)
        filled_total = fill_missing_totals(continuous)
        cleaned = handle_missing_values(filled_total)
        logger.info(
            "Preprocessing complete. raw_shape=%s processed_shape=%s",
            df.shape,
            cleaned.shape,
        )
        return PreprocessingResult(raw=df, processed=cleaned)
    except Exception:
        logger.exception("Preprocessing failed.")
        raise

