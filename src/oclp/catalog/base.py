"""Errors shared by OCLP record resolver implementations."""


class CatalogResolutionError(ValueError):
    """A record reference could not be safely resolved."""


class RecordNotFoundError(CatalogResolutionError):
    """No stored record matches the requested reference."""


class AmbiguousRecordReferenceError(CatalogResolutionError):
    """An ID-only reference matches multiple immutable record revisions."""


class CatalogIntegrityError(CatalogResolutionError):
    """A stored record does not satisfy an integrity-bound reference."""
