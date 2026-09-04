"""Local execution environment for the bike-demand demonstration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DemoEnvironment:
    """Resolve local, ignored storage locations for one demonstration run."""

    project_root: Path

    @classmethod
    def default(cls) -> DemoEnvironment:
        """Create an environment rooted at this example project."""

        return cls(project_root=Path(__file__).resolve().parents[2])

    @property
    def data_root(self) -> Path:
        """Return the ignored root for downloaded and generated data."""

        return self.project_root / "data"

    @property
    def oclp_root(self) -> Path:
        """Return the local directory holding canonical OCLP records."""

        return self.data_root / "oclp-0.3"

    @property
    def catalog_path(self) -> Path:
        """Return the local DuckDB index for the OCLP record directory."""

        return self.oclp_root / "catalog.duckdb"

    @property
    def mlflow_root(self) -> Path:
        """Return the local MLflow metadata and artifact area."""

        return self.data_root / "mlflow"

    def materialization_root(self, materialization_id: str) -> Path:
        """Return the payload directory for one local materialization."""

        return self.data_root / "runs" / materialization_id

    def inference_root(self, request_id: str) -> Path:
        """Return the local payload directory for one HTTP inference request."""

        return self.data_root / "inference" / request_id

    def prepare(self) -> None:
        """Create local roots before a run writes durable material."""

        self.oclp_root.mkdir(parents=True, exist_ok=True)
        self.mlflow_root.mkdir(parents=True, exist_ok=True)
