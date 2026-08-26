"""Portable manifest profile for immutable partitioned dataset snapshots."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, JsonValue, model_validator

from oclp.models import OclpModel, RecordReference

DATASET_SNAPSHOT_PROFILE = "dataset-snapshot"
DATASET_SNAPSHOT_PROFILE_VERSION = "0.1.0-draft"


class DatasetSnapshotBinding(OclpModel):
    """The value carried under an Artifact's ``profiles.dataset-snapshot`` key."""

    version: Literal["0.1.0-draft"]


class DatasetSnapshotPartition(OclpModel):
    """One named partition bound to an exact Artifact record."""

    name: str = Field(min_length=1)
    artifact: RecordReference
    values: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def artifact_reference_is_content_bound(self) -> DatasetSnapshotPartition:
        if self.artifact.digest is None:
            raise ValueError(
                "dataset snapshot partitions must include an artifact digest"
            )
        return self


class DatasetSnapshotManifest(OclpModel):
    """Canonical metadata for one immutable logical dataset version."""

    oclp_profile: Literal["dataset-snapshot"] = DATASET_SNAPSHOT_PROFILE
    oclp_profile_version: Literal["0.1.0-draft"] = DATASET_SNAPSHOT_PROFILE_VERSION
    dataset_id: str = Field(min_length=1)
    data_format: str = Field(min_length=1)
    partitions: tuple[DatasetSnapshotPartition, ...] = Field(min_length=1)
    parent: RecordReference | None = None
    annotations: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def partitions_are_unique_and_sorted(self) -> DatasetSnapshotManifest:
        names = [partition.name for partition in self.partitions]
        if len(names) != len(set(names)):
            raise ValueError("dataset snapshot partition names must be unique")
        if names != sorted(names):
            raise ValueError("dataset snapshot partitions must be sorted by name")
        if self.parent is not None and self.parent.digest is None:
            raise ValueError("dataset snapshot parents must include a record digest")
        return self
