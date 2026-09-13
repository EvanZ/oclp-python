"""Bike-demand metadata used only to join OCLP observations to MLflow.

The portable ``lifecycle`` profile owns the release-cycle identity. This
application binding deliberately carries only the MLflow-specific parent run
needed by the optional MLflow projection.
"""

from __future__ import annotations

from collections.abc import Mapping

from oclp.models import OclpModel, ProfileBindings
from pydantic import Field

MLFLOW_EXPERIMENT_NAME = "oclp-bike-demand-service"
MLFLOW_PARENT_PROFILE = "bike_demand.mlflow-parent"
MLFLOW_PARENT_PROFILE_VERSION = "1"


class MlflowParentProfile(OclpModel):
    """The application-owned parent needed by the MLflow adapter."""

    version: str = MLFLOW_PARENT_PROFILE_VERSION
    mlflow_parent_run_id: str = Field(min_length=1)


def mlflow_parent_profiles(*, mlflow_parent_run_id: str) -> ProfileBindings:
    """Return the app binding that attaches one OCLP run to its MLflow parent."""

    return {
        MLFLOW_PARENT_PROFILE: MlflowParentProfile(
            mlflow_parent_run_id=mlflow_parent_run_id,
        ).model_dump(mode="json")
    }


def mlflow_parent_profile_from_profiles(
    profiles: Mapping[str, Mapping[str, object]] | None,
) -> MlflowParentProfile:
    """Read a validated MLflow parent binding from a released ArtifactSet."""

    if profiles is None:
        raise ValueError("release is missing the bike-demand MLflow parent profile")
    value = profiles.get(MLFLOW_PARENT_PROFILE)
    if value is None:
        raise ValueError("release is missing the bike-demand MLflow parent profile")
    return MlflowParentProfile.model_validate(value)
