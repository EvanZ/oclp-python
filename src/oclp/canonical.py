"""RFC 8785 canonical JSON serialization and record digests."""

from __future__ import annotations

import hashlib

import rfc8785

from oclp.models import CoreRecord, Digest, OclpModel


def canonical_json_bytes(record: OclpModel) -> bytes:
    """Serialize an OCLP record or profile value using JCS."""
    payload = record.model_dump(mode="json", exclude_none=True)
    if isinstance(record, CoreRecord) and record.profiles is None:
        payload["profiles"] = None
    return rfc8785.dumps(payload)


def record_digest(record: OclpModel) -> Digest:
    return Digest(value=hashlib.sha256(canonical_json_bytes(record)).hexdigest())
