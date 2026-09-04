"""Callable-bound Computation declarations for explicit OCLP instrumentation."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import wraps
from types import UnionType
from typing import (
    Any,
    Literal,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import Field, JsonValue, model_validator

from oclp.artifacts import ArtifactType
from oclp.canonical import canonical_json_bytes
from oclp.evidence import evidence_implementation
from oclp.models import (
    Computation,
    Implementation,
    ImplementationSource,
    OclpModel,
    ParameterDefinition,
    PortDefinition,
    ProfileBindings,
)

CallableT = TypeVar("CallableT", bound=Callable[..., object])
_COMPUTATION_TEMPLATE_ATTRIBUTE = "__oclp_computation_template__"
_COMPUTATION_INPUT_ARTIFACTS_ATTRIBUTE = "__oclp_input_artifact_types__"


@dataclass(frozen=True)
class ManyArtifacts:
    """SDK input declaration for a port that consumes many same-kind Artifacts.

    ``many(CsvArtifact)`` is a concise Python declaration for a portable
    ``PortDefinition(cardinality="many")``.  The decorated function still
    receives its annotated in-memory collection, such as
    ``tuple[pandas.DataFrame, ...]``.
    """

    artifact_type: type[ArtifactType]


def many(artifact_type: type[ArtifactType]) -> ManyArtifacts:
    """Declare that one Computation input port consumes many Artifacts."""

    if not isinstance(artifact_type, type) or not issubclass(
        artifact_type, ArtifactType
    ):
        raise TypeError("many() requires a concrete Artifact type")
    return ManyArtifacts(artifact_type=artifact_type)


@dataclass(frozen=True)
class ArtifactSetInput:
    """SDK declaration for one immutable ArtifactSet input.

    The OCLP Execution records the ArtifactSet itself as its input.  The
    member declarations are runtime validation: they identify the named
    Artifacts a callable requires from that set without assigning the
    collection a made-up payload media type.
    """

    members: Mapping[str, type[ArtifactType]]

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError("ArtifactSet inputs must declare at least one member")
        normalized = dict(self.members)
        for name, artifact_type in normalized.items():
            if not isinstance(name, str) or not name:
                raise ValueError("ArtifactSet member names must be non-empty strings")
            if not isinstance(artifact_type, type) or not issubclass(
                artifact_type, ArtifactType
            ):
                raise TypeError(
                    "ArtifactSet input members must declare concrete Artifact types"
                )
        object.__setattr__(self, "members", normalized)


def artifact_set_input(
    members: Mapping[str, type[ArtifactType]],
) -> ArtifactSetInput:
    """Declare the named members required from one ArtifactSet input."""

    return ArtifactSetInput(members=members)


class ComputationTemplate(OclpModel):
    """Static Computation metadata attached to a callable by ``@computation``.

    A template intentionally omits ``Implementation.source``. The selected
    Git, Artifact, service, or opaque source basis is a publication-time fact
    supplied to :func:`computation_record`. Its ``id`` is SDK metadata, already
    an opaque UUID derived from the decorator's application declaration key;
    it is not itself an emitted Core Computation record.
    """

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    profiles: ProfileBindings | None = None
    annotations: dict[str, JsonValue] = Field(default_factory=dict)
    implementation_kind: Literal["python-callable"] = "python-callable"
    input_ports: tuple[PortDefinition, ...] = ()
    output_ports: tuple[PortDefinition, ...] = ()
    parameter_definitions: tuple[ParameterDefinition, ...] = ()
    output_artifacts: dict[str, ArtifactType] = Field(default_factory=dict)
    required_evaluators: tuple[Callable[..., object], ...] | None = None

    @model_validator(mode="after")
    def port_names_are_unique(self) -> ComputationTemplate:
        for ports in (self.input_ports, self.output_ports):
            names = [port.name for port in ports]
            if len(names) != len(set(names)):
                raise ValueError("port names must be unique within each direction")
        parameter_names = [parameter.name for parameter in self.parameter_definitions]
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("computation parameter names must be unique")
        if set(parameter_names).intersection(port.name for port in self.input_ports):
            raise ValueError(
                "computation parameter names must not overlap input port names"
            )
        identifiers = [
            _callable_locator(evaluator) for evaluator in self.required_evaluators or ()
        ]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("required Evidence evaluators must be unique")
        return self


def computation(
    *,
    id: str,
    name: str,
    input_ports: tuple[PortDefinition, ...] = (),
    inputs: Mapping[
        str, type[ArtifactType] | ManyArtifacts | ArtifactSetInput
    ]
    | None = None,
    output_ports: tuple[PortDefinition, ...] = (),
    outputs: Mapping[str, ArtifactType] | None = None,
    requires: tuple[Callable[..., object], ...] | None = None,
    profiles: dict[str, dict[str, JsonValue]] | None = None,
    annotations: dict[str, JsonValue] | None = None,
) -> Callable[[CallableT], CallableT]:
    """Declare a callable's Computation contract and optional persisted outputs.

    Outside an active :class:`oclp.runtime.OclpRun`, the decorated callable
    keeps normal Python behavior. Within one, declared ``outputs`` turn the
    function's return value into published Artifact bindings automatically.
    ``inputs`` is the concise SDK declaration for Artifact inputs. Its
    mapping key is the callable parameter and OCLP port name; its value is the
    expected representation class, such as :class:`oclp.artifacts.CsvArtifact`, or
    ``many(CsvArtifact)`` for an Artifact collection. The SDK derives the
    portable :class:`~oclp.models.PortDefinition` media-type and cardinality
    contract from that declaration while the callable annotation remains the
    actual in-memory value delivered to its body.

    For multiple outputs, every port name must match a field on the returned
    object (or a key in a returned mapping). A single output persists the
    return value itself.

    Each output uses an explicit concrete :class:`ArtifactType`, so persistence
    and its durable representation are never inferred merely from a Python
    return type.
    """

    opaque_id = _opaque_computation_id(id)
    if input_ports and inputs is not None:
        raise ValueError("declare either input_ports or inputs, not both")
    input_ports, input_artifact_types = _input_artifact_ports(inputs, input_ports)

    output_artifacts = _output_artifacts(outputs)
    if output_ports and output_artifacts:
        raise ValueError("declare either output_ports or outputs, not both")
    if output_artifacts:
        output_ports = tuple(
            PortDefinition(
                name=name,
                media_types=(spec.media_type,),
            )
            for name, spec in output_artifacts.items()
        )
    # Validate the decorator arguments immediately, before Python applies the
    # decorator to a callable. Parameter definitions are the only portion of
    # the contract that requires the callable signature itself.
    ComputationTemplate(
        id=opaque_id,
        name=name,
        input_ports=input_ports,
        output_ports=output_ports,
        output_artifacts=output_artifacts,
        required_evaluators=requires,
        profiles=profiles,
        annotations=annotations or {},
    )

    def decorate(function: CallableT) -> CallableT:
        if not callable(function):
            raise TypeError("@computation can only decorate a callable")
        if getattr(function, _COMPUTATION_TEMPLATE_ATTRIBUTE, None) is not None:
            raise ValueError("a callable can have only one OCLP Computation template")
        _validate_input_port_parameters(function, input_ports)
        template = ComputationTemplate(
            id=opaque_id,
            name=name,
            input_ports=input_ports,
            output_ports=output_ports,
            parameter_definitions=_infer_parameter_definitions(function, input_ports),
            output_artifacts=output_artifacts,
            required_evaluators=requires,
            profiles=profiles,
            annotations=annotations or {},
        )
        try:
            setattr(function, _COMPUTATION_TEMPLATE_ATTRIBUTE, template)
            setattr(
                function,
                _COMPUTATION_INPUT_ARTIFACTS_ATTRIBUTE,
                input_artifact_types,
            )
        except (AttributeError, TypeError) as error:
            raise TypeError(
                "@computation requires a callable that accepts attached metadata"
            ) from error

        @wraps(function)
        def observed(*args: object, **kwargs: object) -> object:
            # Local import keeps static template declaration independent of the
            # runtime and avoids a module-import cycle.
            from oclp.runtime import active_run

            run = active_run()
            current_template = computation_template(observed)
            if run is None or (
                not current_template.output_artifacts
                and not current_template.input_ports
                and not current_template.parameter_definitions
                and not current_template.required_evaluators
            ):
                return function(*args, **kwargs)
            return run.invoke(function, current_template, args, kwargs)

        setattr(observed, _COMPUTATION_TEMPLATE_ATTRIBUTE, template)
        setattr(
            observed,
            _COMPUTATION_INPUT_ARTIFACTS_ATTRIBUTE,
            input_artifact_types,
        )
        return cast(CallableT, observed)

    return decorate


def _opaque_computation_id(value: str) -> str:
    """Normalize a UUID or derive opaque SDK template metadata from a key."""

    try:
        return str(UUID(value))
    except ValueError:
        return str(uuid5(NAMESPACE_URL, f"oclp:record-id:{value}"))

def _input_artifact_ports(
    inputs: Mapping[
        str, type[ArtifactType] | ManyArtifacts | ArtifactSetInput
    ]
    | None,
    input_ports: tuple[PortDefinition, ...],
) -> tuple[
    tuple[PortDefinition, ...],
    dict[str, type[ArtifactType] | ManyArtifacts | ArtifactSetInput],
]:
    """Derive portable input ports from concise SDK Artifact representation types."""

    if inputs is None:
        return input_ports, {}
    if not isinstance(inputs, Mapping):
        raise TypeError(
            "computation inputs must map port names to concrete Artifact types"
        )

    artifact_types = dict(inputs)
    ports: list[PortDefinition] = []
    for name, declaration in artifact_types.items():
        if not isinstance(name, str) or not name:
            raise ValueError("computation input names must be non-empty strings")
        if isinstance(declaration, ArtifactSetInput):
            ports.append(PortDefinition(name=name))
            continue
        cardinality: Literal["one", "many"] = "one"
        artifact_type = declaration
        if isinstance(declaration, ManyArtifacts):
            cardinality = "many"
            artifact_type = declaration.artifact_type
        if not isinstance(artifact_type, type) or not issubclass(
            artifact_type, ArtifactType
        ):
            raise TypeError(
                f"computation input {name!r} must declare a concrete Artifact type"
            )
        if not artifact_type.media_types:
            raise ValueError(
                f"Artifact type {artifact_type.__name__!r} must declare "
                "at least one media type"
            )
        ports.append(
            PortDefinition(
                name=name,
                cardinality=cardinality,
                media_types=artifact_type.media_types,
            )
        )
    return tuple(ports), artifact_types


def _output_artifacts(
    outputs: Mapping[str, ArtifactType] | None,
) -> dict[str, ArtifactType]:
    if outputs is None:
        return {}
    if not isinstance(outputs, Mapping):
        raise TypeError(
            "computation outputs must be a mapping of port names to concrete "
            "Artifact values"
        )
    specs = dict(outputs)
    if any(not isinstance(name, str) or not name for name in specs):
        raise ValueError("computation output names must be non-empty strings")
    if any(not isinstance(spec, ArtifactType) for spec in specs.values()):
        raise TypeError(
            "computation output mappings must contain concrete Artifact values"
        )
    unnamed = [name for name, spec in specs.items() if spec.name is None]
    if unnamed:
        rendered = ", ".join(repr(name) for name in unnamed)
        raise ValueError(
            "computation outputs require application-supplied ArtifactType names: "
            f"{rendered}"
        )
    return specs


def _validate_input_port_parameters(
    function: Callable[..., object], ports: tuple[PortDefinition, ...]
) -> None:
    """Require every declared input port to name a real callable parameter."""

    parameters = {
        parameter.name
        for parameter in inspect.signature(function).parameters.values()
        if parameter.name not in {"self", "cls"}
    }
    missing = [port.name for port in ports if port.name not in parameters]
    if missing:
        function_name = getattr(function, "__qualname__", repr(function))
        raise ValueError(
            f"Computation input ports {missing!r} do not match parameters on "
            f"{function_name!r}"
        )


def _infer_parameter_definitions(
    function: Callable[..., object], input_ports: tuple[PortDefinition, ...]
) -> tuple[ParameterDefinition, ...]:
    """Infer portable parameter declarations from ordinary callable arguments.

    Input ports are the durable Artifact boundary. Every other declared
    ordinary argument with a JSON-representable annotation is an Execution
    parameter, including an effective default. Local implementation plumbing
    such as ``pathlib.Path`` is deliberately excluded: it neither identifies
    durable input nor belongs in a reproducible computation interface.
    """

    try:
        hints = get_type_hints(function)
    except (NameError, TypeError):
        hints = getattr(function, "__annotations__", {})

    input_names = {port.name for port in input_ports}
    definitions: list[ParameterDefinition] = []
    for parameter in inspect.signature(function).parameters.values():
        if parameter.name in {"self", "cls"} or parameter.name in input_names:
            continue
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        schema = _json_schema_for_annotation(hints.get(parameter.name))
        if schema is None:
            continue
        if parameter.default is inspect.Parameter.empty:
            required = True
        else:
            default = _json_value(parameter.default)
            if default is None and parameter.default is not None:
                # A non-JSON default identifies local implementation behavior,
                # not a portable Execution parameter.
                continue
            schema = {**schema, "default": default}
            required = False
        definitions.append(
            ParameterDefinition(
                name=parameter.name,
                schema=schema,
                required=required,
            )
        )
    return tuple(definitions)


def _json_schema_for_annotation(annotation: object) -> dict[str, JsonValue] | None:
    """Translate the intentionally small, JSON-safe Python annotation subset.

    ``None`` means the annotation is an SDK-local value rather than a portable
    parameter type. The core record carries JSON Schema, not this inference
    algorithm, so other language implementations remain free to declare the
    same contract directly.
    """

    if annotation is None or annotation is type(None):
        return {"type": "null"}
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}
    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}

    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin is Literal:
        values = list(arguments)
        if not all(_json_value(value) is not None or value is None for value in values):
            return None
        schema: dict[str, JsonValue] = {"enum": values}
        value_types = {type(value) for value in values if value is not None}
        if len(value_types) == 1:
            value_type = next(iter(value_types))
            primitive_schema = _json_schema_for_annotation(value_type)
            if primitive_schema and "type" in primitive_schema:
                schema = {**primitive_schema, "enum": values}
        return schema
    if origin in {list, tuple, set, frozenset}:
        item_schema = _json_schema_for_annotation(arguments[0]) if arguments else {}
        if item_schema is None:
            return None
        return {"type": "array", "items": item_schema}
    if origin is dict:
        if arguments and arguments[0] is not str:
            return None
        value_schema = (
            _json_schema_for_annotation(arguments[1]) if len(arguments) > 1 else {}
        )
        if value_schema is None:
            return None
        return {"type": "object", "additionalProperties": value_schema}
    if origin in {Union, UnionType}:
        variants = [_json_schema_for_annotation(argument) for argument in arguments]
        if any(variant is None for variant in variants):
            return None
        return {"anyOf": cast(list[JsonValue], variants)}
    return None


def _json_value(value: object) -> JsonValue | None:
    """Return a JSON-compatible value or ``None`` for unsupported values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return cast(JsonValue, value)
    if isinstance(value, (list, tuple)):
        values = [_json_value(item) for item in value]
        if any(item is None for item in values):
            return None
        return cast(JsonValue, values)
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        values = {key: _json_value(item) for key, item in value.items()}
        if any(item is None for item in values.values()):
            return None
        return cast(JsonValue, values)
    return None


def computation_template(function: Callable[..., object]) -> ComputationTemplate:
    """Return the static OCLP Computation metadata attached to ``function``."""

    template = getattr(function, _COMPUTATION_TEMPLATE_ATTRIBUTE, None)
    if not isinstance(template, ComputationTemplate):
        name = getattr(function, "__qualname__", repr(function))
        raise ValueError(f"callable {name!r} has no OCLP Computation template")
    return template


def computation_input_artifact_types(
    function: Callable[..., object],
) -> dict[str, type[ArtifactType] | ManyArtifacts | ArtifactSetInput]:
    """Return SDK Artifact representation requirements declared by a callable.

    The mapping is SDK-only metadata. ``computation_record()`` emits its
    language-neutral counterpart as ``Computation.input_ports``.
    """

    declared = getattr(function, _COMPUTATION_INPUT_ARTIFACTS_ATTRIBUTE, {})
    if not isinstance(declared, dict) or any(
        not isinstance(name, str) or not _is_artifact_input_declaration(declaration)
        for name, declaration in declared.items()
    ):
        raise ValueError("callable has invalid OCLP Artifact input metadata")
    return dict(declared)


def _is_artifact_input_declaration(
    declaration: object,
) -> bool:
    if isinstance(declaration, ArtifactSetInput):
        return True
    if isinstance(declaration, ManyArtifacts):
        declaration = declaration.artifact_type
    return isinstance(declaration, type) and issubclass(declaration, ArtifactType)


def computation_record(
    function: Callable[..., object],
    *,
    source: ImplementationSource,
) -> Computation:
    """Materialize the decorated callable as a source-bound OCLP Computation."""

    template = computation_template(function)
    # The decorator's UUID identifies SDK declaration metadata. A Core
    # Computation additionally binds an immutable implementation source, so a
    # different selected commit (or artifact/service source) is a distinct
    # record with a distinct opaque UUID. The UUID5 is reproducible for the
    # same declaration+source without exposing a semantic Core identifier.
    record_id = str(
        uuid5(
            NAMESPACE_URL,
            "oclp:computation-record:"
            f"{template.id}:{canonical_json_bytes(source).decode('utf-8')}",
        )
    )
    return Computation(
        id=record_id,
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
        parameter_definitions=template.parameter_definitions,
        required_evidence=tuple(
            evidence_implementation(evaluator, source=source)
            for evaluator in template.required_evaluators or ()
        )
        or None,
    )


def _callable_locator(function: Callable[..., object]) -> str:
    module = getattr(function, "__module__", None)
    qualname = getattr(function, "__qualname__", None)
    if not isinstance(module, str) or not module or not isinstance(qualname, str):
        raise TypeError("decorated callable must expose __module__ and __qualname__")
    if not qualname or "<locals>" in qualname:
        raise ValueError("decorated callable must have a stable module-level locator")
    return f"{module}.{qualname}"
