"""Filesystem locations for one local bike-demand demonstration run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DemoPaths:
    """Resolve durable local paths without placing generated data in source control."""

    project_root: Path

    @classmethod
    def default(cls) -> DemoPaths:
        """Create paths rooted at this example project."""

        return cls(project_root=Path(__file__).resolve().parents[2])

    @property
    def data_root(self) -> Path:
        """Return the ignored root for downloaded and generated data."""

        return self.project_root / "data"

    @property
    def oclp_root(self) -> Path:
        """Return the simple on-disk OCLP record directory."""

        return self.data_root / "oclp"

    @property
    def catalog_path(self) -> Path:
        """Return the local DuckDB index for the OCLP record directory."""

        return self.oclp_root / "catalog.duckdb"

    @property
    def mlflow_root(self) -> Path:
        """Return the local MLflow metadata and report area."""

        return self.data_root / "mlflow"

    def run_root(self, run_id: str) -> Path:
        """Return the artifact payload directory for one immutable observation."""

        return self.data_root / "runs" / run_id

    def prepare(self) -> None:
        """Create local roots before a pipeline writes any durable material."""

        self.oclp_root.mkdir(parents=True, exist_ok=True)
        self.mlflow_root.mkdir(parents=True, exist_ok=True)
