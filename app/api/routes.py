"""
API routes (Phase 5).

Provides:
- health endpoint (handled in app/main.py)
- state-wise forecast endpoint backed by the saved best model
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Path
from pydantic import BaseModel, Field

from app.config import settings
from app.services.forecasting import forecast_for_state
from app.services.preprocessing import load_dataset, run_preprocessing
from app.utils.logger import get_logger


logger = get_logger(__name__, level=settings.log_level, log_dir=settings.logs_dir)

router = APIRouter(tags=["forecasting"])


class ForecastResponse(BaseModel):
    state: str = Field(..., description="State name")
    forecast_weeks: int = Field(..., description="Forecast horizon (weeks)")
    best_model: str = Field(..., description="Selected best model")
    predictions: List[float] = Field(..., description="Next N predictions")


def _normalize_state(s: str) -> str:
    return str(s).strip().title()


@lru_cache(maxsize=1)
def _available_states() -> set[str]:
    """
    Determine valid states from the dataset (cached).
    """

    raw = load_dataset()
    pre = run_preprocessing(raw).processed
    states = set(pre["State"].astype(str).map(_normalize_state).tolist())
    return {s for s in states if s}


def _predict_for_state(
    *,
    state: str,
    steps: int,
) -> tuple[str, List[float]]:
    """
    Load best model artifact and return predictions for a single state.
    """
    try:
        model_name, preds = forecast_for_state(state=state, steps=steps)
        return model_name, [float(x) for x in preds.tolist()]
    except KeyError:
        raise HTTPException(status_code=404, detail="State not found in trained models")


@router.get(
    "/forecast/{state}",
    summary="Forecast next 8 weeks sales for a state",
    response_model=ForecastResponse,
)
def forecast_state(
    state: str = Path(..., description="State name (trimmed)"),
    weeks: Optional[int] = None,
) -> ForecastResponse:
    """
    Generate the next N weeks forecast for a specific state using the saved best model.
    """

    state = _normalize_state(state)
    if not state:
        raise HTTPException(status_code=400, detail="Invalid state")

    valid_states = _available_states()
    if state not in valid_states:
        raise HTTPException(status_code=400, detail="Invalid state")

    steps = int(weeks or settings.forecast_horizon_weeks)
    if steps <= 0:
        raise HTTPException(status_code=400, detail="Invalid forecast weeks")

    logger.info("Forecast requested for %s (weeks=%s)", state, steps)

    try:
        model_name, preds = _predict_for_state(state=state, steps=steps)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Missing model. Run training pipeline first.")

    if not preds:
        raise HTTPException(status_code=500, detail="Empty prediction")

    logger.info("Forecast served for %s model=%s points=%s", state, model_name, len(preds))
    return ForecastResponse(
        state=state,
        forecast_weeks=steps,
        best_model=model_name,
        predictions=preds,
    )

