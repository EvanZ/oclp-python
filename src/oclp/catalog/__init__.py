"""Optional catalog and reference-resolution interfaces."""

from oclp.catalog.base import (
    AmbiguousRecordReferenceError,
    CatalogIntegrityError,
    CatalogResolutionError,
    RecordNotFoundError,
)

__all__ = [
    "AmbiguousRecordReferenceError",
    "CatalogIntegrityError",
    "CatalogResolutionError",
    "RecordNotFoundError",
]
