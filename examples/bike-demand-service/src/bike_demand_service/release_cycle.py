"""Application-owned release-cycle profile helpers.

This module deliberately belongs to the bike-demand application, not the
generic SDK.  OCLP identifies individual records by UUID; a release cycle is
the application's higher-level grouping across independently scheduled jobs.
"""

from __future__ import annotations

from collections.abc import Mapping

from oclp.models import OclpModel, ProfileBindings
from pydantic import Field

RELEASE_CYCLE_PROFILE = "bike_demand"
RELEASE_CYCLE_PROFILE_VERSION = "1"
MLFLOW_EXPERIMENT_NAME = "oclp-bike-demand-service"


class ReleaseCycleProfile(OclpModel):
    """Durable application facts that associate records with one cycle."""

    version: str = RELEASE_CYCLE_PROFILE_VERSION
    release_cycle_id: str = Field(min_length=1)
    mlflow_parent_run_id: str = Field(min_length=1)


def release_cycle_profiles(
    *,
    release_cycle_id: str,
    mlflow_parent_run_id: str,
) -> ProfileBindings:
    """Return the application profile binding for a release-cycle record."""

    return {
        RELEASE_CYCLE_PROFILE: ReleaseCycleProfile(
            release_cycle_id=release_cycle_id,
            mlflow_parent_run_id=mlflow_parent_run_id,
        ).model_dump(mode="json")
    }


def release_cycle_profile_from_profiles(
    profiles: Mapping[str, Mapping[str, object]] | None,
) -> ReleaseCycleProfile:
    """Validate the cycle binding carried by a released ArtifactSet."""

    if profiles is None:
        raise ValueError("release is missing the bike_demand release-cycle profile")
    value = profiles.get(RELEASE_CYCLE_PROFILE)
    if value is None:
        raise ValueError("release is missing the bike_demand release-cycle profile")
    return ReleaseCycleProfile.model_validate(value)
