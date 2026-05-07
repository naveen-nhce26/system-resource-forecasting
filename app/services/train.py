"""
Automated training + evaluation pipeline (Phase 4).

run_pipeline()
→ load data
→ preprocess
→ feature engineer
→ train SARIMA / Prophet / XGBoost / LSTM (state-wise)
→ evaluate all models (RMSE/MAE/MAPE)
→ compare + select best model
→ save best model artifact(s)

No random splitting is used anywhere (chronology preserved).
"""

from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd

from app.config import settings
from app.services.evaluate import ModelMetrics, leaderboard as make_leaderboard
from app.services.evaluate import select_best_model
from app.services.preprocessing import load_dataset, run_preprocessing
from app.services.feature_engineering import build_features
from app.utils.logger import get_logger


logger = get_logger(__name__, level=settings.log_level, log_dir=settings.logs_dir)


@dataclass(frozen=True)
class PipelineConfig:
    """
    Top-level pipeline configuration.
    """

    horizon: int = settings.forecast_horizon_weeks
    primary_metric: str = "rmse"  # rmse|mae|mape

    # Model configs are stored as primitives to keep imports lightweight.
    # They are converted into model-specific config objects inside train steps.
    sarima_order: tuple[int, int, int] = (1, 1, 1)
    sarima_seasonal_order: tuple[int, int, int, int] = (0, 1, 1, 52)

    prophet_seasonality_mode: str = "additive"
    prophet_changepoint_prior_scale: float = 0.05

    xgb_params: Dict[str, Any] = None  # type: ignore[assignment]

    lstm_window: int = 30
    lstm_units: int = 64
    lstm_learning_rate: float = 1e-3
    lstm_epochs: int = 20
    lstm_batch_size: int = 32
    lstm_verbose: int = 0

    def __post_init__(self) -> None:
        if self.xgb_params is None:
            object.__setattr__(
                self,
                "xgb_params",
                {
                    "n_estimators": 500,
                    "learning_rate": 0.05,
                    "max_depth": 6,
                    "subsample": 0.9,
                    "colsample_bytree": 0.9,
                    "reg_alpha": 0.0,
                    "reg_lambda": 1.0,
                    "random_state": 42,
                    "n_jobs": 0,
                },
            )


@dataclass(frozen=True)
class PipelineArtifacts:
    """
    Outputs of pipeline execution.
    """

    preprocessed: pd.DataFrame
    features: pd.DataFrame
    models: Dict[str, Any]
    metrics: Dict[str, ModelMetrics]
    leaderboard: pd.DataFrame
    best_model_name: str
    saved_path: Path


def _ensure_saved_models_dir() -> Path:
    settings.saved_models_dir.mkdir(parents=True, exist_ok=True)
    return settings.saved_models_dir


def _evaluate_aggregate(per_state_metrics: Dict[str, Dict[str, float]]) -> ModelMetrics:
    # Lazy import to avoid circular dependency; evaluate.py already defines the type.
    import numpy as np

    rmses = [m["rmse"] for m in per_state_metrics.values()]
    maes = [m["mae"] for m in per_state_metrics.values()]
    mapes = [m["mape"] for m in per_state_metrics.values()]
    return ModelMetrics(
        rmse=float(np.mean(rmses)) if rmses else float("nan"),
        mae=float(np.mean(maes)) if maes else float("nan"),
        mape=float(np.mean(mapes)) if mapes else float("nan"),
        n_states=len(per_state_metrics),
    )


def _train_sarima(preprocessed: pd.DataFrame, *, cfg: PipelineConfig) -> tuple[Dict[str, SarimaStateModel], ModelMetrics]:
    from app.models.sarima_model import SarimaConfig, SarimaStateModel
    from app.models.sarima_model import train_and_evaluate_state as train_eval_sarima_state

    per_state: Dict[str, SarimaStateModel] = {}
    per_state_metrics: Dict[str, Dict[str, float]] = {}

    # Practical, lightweight SARIMA for stable execution.
    sarima_cfg = SarimaConfig(
        order=cfg.sarima_order,
        seasonal_order=(0, 0, 0, 0),  # default to non-seasonal for stability
        allow_seasonal=False,
        enforce_stationarity=False,
        enforce_invertibility=False,
        maxiter=50,
    )
    for state, g in preprocessed.groupby("State", sort=False):
        try:
            logger.info("SARIMA training start state=%s rows=%s", state, len(g))
            fitted, metrics = train_eval_sarima_state(g, config=sarima_cfg, val_size=cfg.horizon)
            per_state[str(state)] = SarimaStateModel(state=str(state), config=sarima_cfg, fitted=fitted)
            per_state_metrics[str(state)] = metrics
            logger.info("SARIMA training done state=%s", state)
        except Exception:
            logger.exception("SARIMA failed for state=%s. Skipping state.", state)
            continue

    agg = _evaluate_aggregate(per_state_metrics)
    return per_state, agg


def _train_prophet(preprocessed: pd.DataFrame, *, cfg: PipelineConfig) -> tuple[Dict[str, ProphetStateModel], ModelMetrics]:
    from app.models.prophet_model import ProphetConfig, ProphetStateModel
    from app.models.prophet_model import train_and_evaluate_state as train_eval_prophet_state

    per_state: Dict[str, ProphetStateModel] = {}
    per_state_metrics: Dict[str, Dict[str, float]] = {}

    prophet_cfg = ProphetConfig(
        seasonality_mode=cfg.prophet_seasonality_mode,
        changepoint_prior_scale=cfg.prophet_changepoint_prior_scale,
    )
    for state, g in preprocessed.groupby("State", sort=False):
        model, freq, metrics = train_eval_prophet_state(g, config=prophet_cfg, val_size=cfg.horizon)
        per_state[str(state)] = ProphetStateModel(state=str(state), config=prophet_cfg, model=model, freq=freq)
        per_state_metrics[str(state)] = metrics

    agg = _evaluate_aggregate(per_state_metrics)
    return per_state, agg


def _train_xgboost(features: pd.DataFrame, *, cfg: PipelineConfig) -> tuple[Dict[str, XGBoostStateModel], ModelMetrics]:
    from app.models.xgboost_model import XGBoostConfig, XGBoostStateModel
    from app.models.xgboost_model import train_state as train_xgb_state

    per_state: Dict[str, XGBoostStateModel] = {}
    per_state_metrics: Dict[str, Dict[str, float]] = {}

    xgb_cfg = XGBoostConfig(**cfg.xgb_params)
    for state, g in features.groupby("State", sort=False):
        model, metrics = train_xgb_state(g.sort_values("Date"), config=xgb_cfg, val_size=cfg.horizon)
        per_state[str(state)] = XGBoostStateModel(state=str(state), config=xgb_cfg, model=model)
        per_state_metrics[str(state)] = metrics

    agg = _evaluate_aggregate(per_state_metrics)
    return per_state, agg


def _train_lstm(preprocessed: pd.DataFrame, *, cfg: PipelineConfig) -> tuple[Dict[str, LSTMStateModel], ModelMetrics]:
    from app.models.lstm_model import LSTMConfig, LSTMStateModel
    from app.models.lstm_model import train_state as train_lstm_state

    per_state: Dict[str, LSTMStateModel] = {}
    per_state_metrics: Dict[str, Dict[str, float]] = {}

    lstm_cfg = LSTMConfig(
        window=cfg.lstm_window,
        lstm_units=cfg.lstm_units,
        learning_rate=cfg.lstm_learning_rate,
        epochs=cfg.lstm_epochs,
        batch_size=cfg.lstm_batch_size,
        verbose=cfg.lstm_verbose,
    )
    for state, g in preprocessed.groupby("State", sort=False):
        model, scaler, metrics = train_lstm_state(g.sort_values("Date"), config=lstm_cfg, val_size=cfg.horizon)
        per_state[str(state)] = LSTMStateModel(state=str(state), config=lstm_cfg, model=model, scaler=scaler)
        per_state_metrics[str(state)] = metrics

    agg = _evaluate_aggregate(per_state_metrics)
    return per_state, agg


def save_best_model(
    *,
    best_model_name: str,
    models: Dict[str, Any],
) -> Path:
    """
    Save the selected best model to `app/saved_models/`.

    - Classical/ML models: joblib
    - LSTM: Keras `.keras` file + joblib scaler + metadata
    """

    saved_dir = _ensure_saved_models_dir()

    if best_model_name != "LSTM":
        path = saved_dir / "best_model.joblib"
        payload = {"model_name": best_model_name, "models_by_state": models[best_model_name]}
        joblib.dump(payload, path)
        return path

    # LSTM special handling
    from app.models.lstm_model import LSTMStateModel

    lstm_dir = saved_dir / "best_lstm"
    lstm_dir.mkdir(parents=True, exist_ok=True)

    models_by_state: Dict[str, LSTMStateModel] = models["LSTM"]
    # Save a metadata + scalers via joblib; save keras models per state.
    scaler_payload: Dict[str, Any] = {"model_name": "LSTM", "states": list(models_by_state.keys())}
    for state, sm in models_by_state.items():
        model_path = lstm_dir / f"{state}.keras"
        sm.model.save(model_path)
        scaler_payload[state] = {"config": sm.config, "scaler": sm.scaler}

    meta_path = lstm_dir / "lstm_meta.joblib"
    joblib.dump(scaler_payload, meta_path)
    return lstm_dir


def run_pipeline(*, config: Optional[PipelineConfig] = None) -> PipelineArtifacts:
    """
    Run full automated pipeline and persist the best model.
    """

    cfg = config or PipelineConfig()

    try:
        print("Starting training pipeline...")
        logger.info("Pipeline start. horizon=%s primary_metric=%s", cfg.horizon, cfg.primary_metric)

        print("Loading dataset...")
        raw = load_dataset()
        logger.info("Dataset loaded. shape=%s", raw.shape)

        print("Preprocessing...")
        pre = run_preprocessing(raw).processed
        logger.info("Preprocessing completed. shape=%s", pre.shape)

        print("Feature engineering...")
        feats = build_features(pre).features
        logger.info("Feature engineering completed. shape=%s", feats.shape)

        models: Dict[str, Any] = {}
        metrics: Dict[str, ModelMetrics] = {}

        print("Training SARIMA...")
        sarima_models, sarima_metrics = _train_sarima(pre, cfg=cfg)
        models["SARIMA"] = sarima_models
        metrics["SARIMA"] = sarima_metrics

        print("Training Prophet...")
        prophet_models, prophet_metrics = _train_prophet(pre, cfg=cfg)
        models["Prophet"] = prophet_models
        metrics["Prophet"] = prophet_metrics

        print("Training XGBoost...")
        xgb_models, xgb_metrics = _train_xgboost(feats, cfg=cfg)
        models["XGBoost"] = xgb_models
        metrics["XGBoost"] = xgb_metrics

        print("Training LSTM...")
        lstm_models, lstm_metrics = _train_lstm(pre, cfg=cfg)
        models["LSTM"] = lstm_models
        metrics["LSTM"] = lstm_metrics

        print("Evaluating and ranking models...")
        lb = make_leaderboard(metrics, primary_metric=cfg.primary_metric)
        best = select_best_model(metrics, primary_metric=cfg.primary_metric)

        print(f"Best model selected: {best}")
        logger.info("Leaderboard:\n%s", lb.to_string(index=False) if not lb.empty else "<empty>")

        print("Saving best model...")
        saved_path = save_best_model(best_model_name=best, models=models)

        print(f"Model saved to: {saved_path}")
        logger.info("Pipeline complete. best_model=%s saved_path=%s", best, saved_path)
        return PipelineArtifacts(
            preprocessed=pre,
            features=feats,
            models=models,
            metrics=metrics,
            leaderboard=lb,
            best_model_name=best,
            saved_path=saved_path,
        )
    except Exception:
        print("Training pipeline failed. See error below.", file=sys.stderr)
        traceback.print_exc()
        logger.exception("Pipeline failed.")
        raise


def main(argv: Optional[list[str]] = None) -> None:
    """
    CLI entrypoint for:
    - `python -m app.services.train`
    - `python app/services/train.py`
    """

    parser = argparse.ArgumentParser(description="Run automated training pipeline.")
    parser.add_argument("--horizon", type=int, default=settings.forecast_horizon_weeks, help="Validation horizon (default: 8)")
    parser.add_argument("--metric", type=str, default="rmse", choices=["rmse", "mae", "mape"], help="Primary selection metric")
    args = parser.parse_args(argv)

    cfg = PipelineConfig(horizon=int(args.horizon), primary_metric=str(args.metric))
    run_pipeline(config=cfg)


if __name__ == "__main__":
    # When executed as a script, ensure repo root is on sys.path so `import app...` works.
    # This keeps `python app/services/train.py` working from the repo root.
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    main()

