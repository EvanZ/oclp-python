"""Optional runtime adapters attached to canonical OCLP declarations.

Declaration adapters are intentionally separate from :class:`RunAdapter`.
They may wrap an already-declared callable when a particular host runtime
invokes it, while the canonical OCLP decorator remains the sole source of its
Artifact or Computation semantics.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, Protocol, TypeVar

CallableT = TypeVar("CallableT", bound=Callable[..., object])
DeclarationKind = Literal["artifact", "computation", "artifact_set"]


class DeclarationAdapter(Protocol):
    """Wrap one canonical OCLP declaration for an optional host runtime."""

    def wrap(
        self,
        function: CallableT,
        *,
        kind: DeclarationKind,
    ) -> CallableT:
        """Return a host-aware wrapper while preserving OCLP declaration metadata."""


def apply_declaration_adapters(
    function: CallableT,
    *,
    adapters: tuple[DeclarationAdapter, ...],
    kind: DeclarationKind,
) -> CallableT:
    """Apply declaration adapters in declaration order.

    The core only relies on this structural protocol. Optional integrations
    therefore remain importable without becoming OCLP core dependencies.
    """

    if not isinstance(adapters, tuple):
        raise TypeError("declaration adapters must be a tuple")
    wrapped = function
    for adapter in adapters:
        wrap = getattr(adapter, "wrap", None)
        if not callable(wrap):
            raise TypeError(
                "declaration adapters must implement wrap(function, *, kind)"
            )
        wrapped = wrap(wrapped, kind=kind)
        if not callable(wrapped):
            raise TypeError("declaration adapter wrap() must return a callable")
    return wrapped
