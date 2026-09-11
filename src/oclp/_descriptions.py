"""Private helpers for opt-in callable descriptions."""

from __future__ import annotations

import inspect
from collections.abc import Callable


def resolve_callable_description(
    *,
    description: str | None,
    description_from_docstring: bool,
    function: Callable[..., object],
    decorator: str,
) -> str | None:
    """Choose an explicit description or one normalized callable docstring.

    Callers use this only at decorator declaration time.  Ordinary Python
    comments are intentionally unavailable here: they are not durable runtime
    metadata and must never be guessed as protocol descriptions.
    """

    if not isinstance(description_from_docstring, bool):
        raise TypeError(f"{decorator} description_from_docstring must be a bool")
    if description is not None:
        return description
    if not description_from_docstring:
        return None
    resolved = inspect.getdoc(function)
    if resolved is None or not resolved.strip():
        raise ValueError(
            f"{decorator} description_from_docstring=True requires a non-empty "
            "callable docstring"
        )
    return resolved
