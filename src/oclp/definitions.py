"""Callable-bound Definition declarations for explicit OCLP instrumentation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, TypeVar

from pydantic import Field, JsonValue, model_validator

from oclp.models import (
    ComputationDefinition,
    Implementation,
    ImplementationSource,
    OclpModel,
    PortDefinition,
    ProfileBindings,
)

CallableT = TypeVar("CallableT", bound=Callable[..., object])
_DEFINITION_TEMPLATE_ATTRIBUTE = "__oclp_definition_template__"


class DefinitionTemplate(OclpModel):
    """Static Definition metadata attached to a callable by ``@definition``.

    A template intentionally omits ``Implementation.source``. The exact Git,
    Artifact, service, or opaque source basis is an execution/publication-time
    fact and is supplied to :func:`definition_record`.
    """

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    profiles: ProfileBindings | None = None
    annotations: dict[str, JsonValue] = Field(default_factory=dict)
    implementation_kind: Literal["python-callable"] = "python-callable"
    input_ports: tuple[PortDefinition, ...] = ()
    output_ports: tuple[PortDefinition, ...] = ()

    @model_validator(mode="after")
    def port_names_are_unique(self) -> DefinitionTemplate:
        for ports in (self.input_ports, self.output_ports):
            names = [port.name for port in ports]
            if len(names) != len(set(names)):
                raise ValueError("port names must be unique within each direction")
        return self


def definition(
    *,
    id: str,
    name: str,
    input_ports: tuple[PortDefinition, ...] = (),
    output_ports: tuple[PortDefinition, ...] = (),
    profiles: dict[str, dict[str, JsonValue]] | None = None,
    annotations: dict[str, JsonValue] | None = None,
) -> Callable[[CallableT], CallableT]:
    """Attach static OCLP Definition metadata to an ordinary Python callable.

    The decorator does not wrap the callable, inspect arguments, materialize
    data, or publish records. It keeps Definition declaration colocated with
    the function while leaving Invocation observation explicit to the calling
    application.
    """

    template = DefinitionTemplate(
        id=id,
        name=name,
        input_ports=input_ports,
        output_ports=output_ports,
        profiles=profiles,
        annotations=annotations or {},
    )

    def decorate(function: CallableT) -> CallableT:
        if not callable(function):
            raise TypeError("@definition can only decorate a callable")
        if getattr(function, _DEFINITION_TEMPLATE_ATTRIBUTE, None) is not None:
            raise ValueError("a callable can have only one OCLP Definition template")
        try:
            setattr(function, _DEFINITION_TEMPLATE_ATTRIBUTE, template)
        except (AttributeError, TypeError) as error:
            raise TypeError(
                "@definition requires a callable that accepts attached metadata"
            ) from error
        return function

    return decorate


def definition_template(function: Callable[..., object]) -> DefinitionTemplate:
    """Return the static OCLP Definition metadata attached to ``function``."""

    template = getattr(function, _DEFINITION_TEMPLATE_ATTRIBUTE, None)
    if not isinstance(template, DefinitionTemplate):
        name = getattr(function, "__qualname__", repr(function))
        raise ValueError(f"callable {name!r} has no OCLP Definition template")
    return template


def definition_record(
    function: Callable[..., object],
    *,
    source: ImplementationSource,
) -> ComputationDefinition:
    """Materialize the decorated callable as a source-bound OCLP Definition."""

    template = definition_template(function)
    return ComputationDefinition(
        id=template.id,
        name=template.name,
        profiles=template.profiles,
        annotations=template.annotations,
        implementation=Implementation(
            kind=template.implementation_kind,
            locator=_callable_locator(function),
            source=source,
        ),
        input_ports=template.input_ports,
        output_ports=template.output_ports,
    )


def _callable_locator(function: Callable[..., object]) -> str:
    module = getattr(function, "__module__", None)
    qualname = getattr(function, "__qualname__", None)
    if not isinstance(module, str) or not module or not isinstance(qualname, str):
        raise TypeError("decorated callable must expose __module__ and __qualname__")
    if not qualname or "<locals>" in qualname:
        raise ValueError("decorated callable must have a stable module-level locator")
    return f"{module}.{qualname}"
