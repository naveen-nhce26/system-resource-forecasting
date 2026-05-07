"""
Application configuration for the forecasting backend (Phase 1).

This module keeps environment-ready settings and project paths in one place.
Avoid putting secrets in code; use environment variables in production.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _repo_root() -> Path:
    """
    Resolve repository root as the parent of the `app/` directory.
    """

    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    """
    Environment-ready settings container.
    """

    # App/runtime
    app_name: str = os.getenv("APP_NAME", "timeseries-forecasting-backend")
    environment: str = os.getenv("ENVIRONMENT", "development")  # dev|staging|prod
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # API
    api_prefix: str = os.getenv("API_PREFIX", "/api/v1")

    # Forecasting objective (constants)
    forecast_horizon_weeks: int = int(os.getenv("FORECAST_HORIZON_WEEKS", "8"))

    # Paths
    repo_root: Path = _repo_root()
    data_dir: Path = repo_root / "data"
    logs_dir: Path = repo_root / "app" / "logs"
    saved_models_dir: Path = repo_root / "app" / "saved_models"

    # Dataset
    dataset_filename: str = os.getenv("DATASET_FILENAME", "casestudy.xlsx")

    @property
    def dataset_path(self) -> Path:
        return self.data_dir / self.dataset_filename


settings = Settings()

