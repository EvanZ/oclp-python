"""SDK adapter for the portable lifecycle profile.

The normative specification, JSON Schema, and conformance vectors are owned by
the ``oclp-profiles`` project. This module validates the binding and provides
an opt-in value object for attaching one lifecycle consistently to concrete
OCLP records.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, TypeAlias
from uuid import UUID, uuid4

from oclp.models import OclpModel, ProfileBindings

LIFECYCLE_PROFILE = "lifecycle"
LIFECYCLE_PROFILE_VERSION = "0.3.0-draft"


class LifecycleBinding(OclpModel):
    """The value carried under a concrete record's ``profiles.lifecycle`` key."""

    version: Literal["0.3.0-draft"] = LIFECYCLE_PROFILE_VERSION
    lifecycle_id: UUID


class Lifecycle(OclpModel):
    """One opt-in application-level identity across concrete OCLP records."""

    lifecycle_id: UUID

    @classmethod
    def new(cls) -> Lifecycle:
        """Create a fresh UUID lifecycle identity."""

        return cls(lifecycle_id=uuid4())

    def binding(self) -> LifecycleBinding:
        """Return the validated portable profile binding for this lifecycle."""

        return LifecycleBinding(lifecycle_id=self.lifecycle_id)

    def profile_bindings(self) -> ProfileBindings:
        """Return this lifecycle as an OCLP ``profiles`` map."""

        return {
            LIFECYCLE_PROFILE: self.binding().model_dump(mode="json"),
        }


LifecycleInput: TypeAlias = Lifecycle | UUID | str


def new_lifecycle() -> Lifecycle:
    """Create a fresh opt-in lifecycle for related application work."""

    return Lifecycle.new()


def lifecycle_from_id(lifecycle_id: UUID | str) -> Lifecycle:
    """Rehydrate a lifecycle value from an already selected UUID identity."""

    return Lifecycle(lifecycle_id=lifecycle_id)


def lifecycle_from_profiles(
    profiles: Mapping[str, Mapping[str, object]] | None,
) -> Lifecycle:
    """Read and validate the lifecycle binding carried by one Core record."""

    if profiles is None:
        raise ValueError("record is missing the lifecycle profile")
    value = profiles.get(LIFECYCLE_PROFILE)
    if value is None:
        raise ValueError("record is missing the lifecycle profile")
    binding = LifecycleBinding.model_validate(value)
    return Lifecycle(lifecycle_id=binding.lifecycle_id)


def coerce_lifecycle(value: LifecycleInput) -> Lifecycle:
    """Normalize an explicit lifecycle value without manufacturing one."""

    if isinstance(value, Lifecycle):
        return value
    if isinstance(value, UUID | str):
        return lifecycle_from_id(value)
    raise TypeError("lifecycle must be a Lifecycle, UUID, or UUID string")
