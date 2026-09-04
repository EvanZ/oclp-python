"""Validation adapter for the release-manifest profile binding.

The normative profile specification and schema live in the ``oclp-profiles``
package.  This SDK model only validates the binding that makes a durable
manifest Artifact describe one exact ArtifactSet without making the manifest a
member of that set.
"""

from __future__ import annotations

from typing import Literal

from oclp.models import OclpModel, RecordReference

RELEASE_MANIFEST_PROFILE = "release-manifest"
RELEASE_MANIFEST_PROFILE_VERSION = "0.3.0-draft"


class ReleaseManifestBinding(OclpModel):
    """The value carried under an Artifact's ``profiles.release-manifest`` key."""

    version: Literal["0.3.0-draft"]
    artifact_set: RecordReference
