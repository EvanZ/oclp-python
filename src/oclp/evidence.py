"""Callable-bound Evidence evaluators and ergonomic Evidence materialization."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Literal, TypeVar, cast

from pydantic import Field, JsonValue

from oclp.models import (
    Diagnostic,
    Evidence,
    Implementation,
    ImplementationSource,
    OclpModel,
    ProfileBindings,
    RecordReference,
)

CallableT = TypeVar("CallableT", bound=Callable[..., object])
_EVIDENCE_TEMPLATE_ATTRIBUTE = "__oclp_evidence_template__"


class EvidenceTemplate(OclpModel):
    """Static evaluator metadata attached to a callable by ``@evidence``."""

    name: str = Field(min_length=1)
    profiles: ProfileBindings | None = None
    annotations: dict[str, JsonValue] = Field(default_factory=dict)
    evaluator_kind: Literal["python-callable"] = "python-callable"


def evidence(
    *,
    name: str,
    profiles: dict[str, dict[str, JsonValue]] | None = None,
    annotations: dict[str, JsonValue] | None = None,
) -> Callable[[CallableT], CallableT]:
    """Declare a callable as a source-bound Evidence evaluator.

    The callable remains an ordinary function and returns exactly one OCLP
    Evidence outcome: ``"pass"``, ``"fail"``, or ``"error"``.
    :func:`evaluate_evidence` turns that outcome into an Evidence record bound
    to the exact evaluator implementation and Execution.
    """

    template = EvidenceTemplate(
        name=name,
        profiles=profiles,
        annotations=annotations or {},
    )

    def decorate(function: CallableT) -> CallableT:
        if not callable(function):
            raise TypeError("@evidence can only decorate a callable")
        if getattr(function, _EVIDENCE_TEMPLATE_ATTRIBUTE, None) is not None:
            raise ValueError("a callable can have only one OCLP Evidence template")
        try:
            setattr(function, _EVIDENCE_TEMPLATE_ATTRIBUTE, template)
        except (AttributeError, TypeError) as error:
            raise TypeError(
                "@evidence requires a callable that accepts attached metadata"
            ) from error
        return function

    return decorate


def evidence_template(function: Callable[..., object]) -> EvidenceTemplate:
    """Return the static evaluator metadata attached to ``function``."""

    template = getattr(function, _EVIDENCE_TEMPLATE_ATTRIBUTE, None)
    if not isinstance(template, EvidenceTemplate):
        name = getattr(function, "__qualname__", repr(function))
        raise ValueError(f"callable {name!r} has no OCLP Evidence template")
    return template


def evidence_implementation(
    function: Callable[..., object], *, source: ImplementationSource
) -> Implementation:
    """Bind a decorated evaluator to the source observed for one execution."""

    template = evidence_template(function)
    return Implementation(
        kind=template.evaluator_kind,
        locator=_callable_locator(function),
        source=source,
    )


def evaluate_evidence(
    function: Callable[..., object],
    *args: object,
    subject: RecordReference,
    source: ImplementationSource,
    id: str,
    observed_at: datetime,
    name: str | None = None,
    profiles: ProfileBindings | None = None,
    evaluation_error: Exception | None = None,
    **kwargs: object,
) -> Evidence:
    """Evaluate a decorated callable and materialize its Evidence claim.

    Evaluator exceptions and invalid outcomes become durable ``error`` Evidence,
    allowing an Execution to report every required gate in one run.
    """

    template = evidence_template(function)
    evaluator = evidence_implementation(function, source=source)
    if evaluation_error is not None:
        return _error_evidence(
            id=id,
            name=name or template.name,
            subject=subject,
            evaluator=evaluator,
            observed_at=observed_at,
            error=evaluation_error,
            profiles=_merged_profiles(template.profiles, profiles),
        )
    try:
        result = function(*args, **kwargs)
        if result not in ("pass", "fail", "error"):
            raise TypeError(
                f"Evidence evaluator {evaluator.locator} must return "
                "'pass', 'fail', or 'error'"
            )
    except Exception as error:
        return _error_evidence(
            id=id,
            name=name or template.name,
            subject=subject,
            evaluator=evaluator,
            observed_at=observed_at,
            error=error,
            profiles=_merged_profiles(template.profiles, profiles),
        )
    outcome = cast(Literal["pass", "fail", "error"], result)
    diagnostic = _outcome_diagnostic(outcome, template.name)
    return Evidence(
        id=id,
        name=name or template.name,
        subject=subject,
        evaluator=evaluator,
        outcome=outcome,
        observed_at=observed_at,
        diagnostic=diagnostic,
        profiles=_merged_profiles(template.profiles, profiles),
    )


def _outcome_diagnostic(
    outcome: Literal["pass", "fail", "error"],
    evaluator_name: str,
) -> Diagnostic | None:
    """Explain non-passing evaluator outcomes without extra application ceremony."""

    if outcome == "pass":
        return None
    if outcome == "fail":
        return Diagnostic(
            code="oclp/evidence-failed",
            message=f"{evaluator_name} reported fail.",
            stage="validation",
        )
    return Diagnostic(
        code="oclp/evidence-error-outcome",
        message=f"{evaluator_name} reported error.",
        stage="validation",
    )


def _error_evidence(
    *,
    id: str,
    name: str | None,
    subject: RecordReference,
    evaluator: Implementation,
    observed_at: datetime,
    error: Exception,
    profiles: ProfileBindings | None,
) -> Evidence:
    return Evidence(
        id=id,
        name=name,
        subject=subject,
        evaluator=evaluator,
        outcome="error",
        observed_at=observed_at,
        diagnostic=Diagnostic(
            code="oclp/evidence-evaluator-error",
            message=f"{type(error).__name__}: {error}",
            stage="evaluation",
        ),
        profiles=profiles,
    )


def _merged_profiles(
    declared: ProfileBindings | None,
    runtime: ProfileBindings | None,
) -> ProfileBindings | None:
    """Merge static evaluator profiles with one active runtime context."""

    if declared is None:
        return runtime
    if runtime is None:
        return declared
    merged: ProfileBindings = dict(declared)
    for profile_id, value in runtime.items():
        existing = merged.get(profile_id)
        if existing is not None and existing != value:
            raise ValueError(
                f"Evidence profile {profile_id!r} conflicts with active runtime"
            )
        merged[profile_id] = value
    return merged


def _callable_locator(function: Callable[..., object]) -> str:
    module = getattr(function, "__module__", None)
    qualname = getattr(function, "__qualname__", None)
    if not isinstance(module, str) or not module or not isinstance(qualname, str):
        raise TypeError("decorated callable must expose __module__ and __qualname__")
    if not qualname or "<locals>" in qualname:
        raise ValueError("decorated callable must have a stable module-level locator")
    return f"{module}.{qualname}"
