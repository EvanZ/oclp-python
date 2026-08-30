# Instrument a computation

This guide shows the Python SDK's implementation pattern **as it exists
today**. OCLP does not run your function or automatically discover its
dependencies. Your application decides the computation boundary, materializes
the exact data it used and produced, then publishes an immutable observation
of that work.

The canonical field contracts are in the [OCLP specification](https://evanz.github.io/open-computation-lifecycle/protocol/specification/).
This page explains how to apply them from Python.

## What the SDK does today

The SDK provides strict record models, canonical JSON and digests, graph
validation, profile helpers, an optional local `DuckdbCatalog`, and a narrow
`@oclp.definition` declaration decorator. It does **not** include an
`@oclp.task` runtime tracer or a generic automatic-instrumentation helper.

That is deliberate for now. A decorator can observe function entry and exit,
but it cannot determine, without application policy:

- which files, rows, objects, model package, or remote response are the exact
  input Artifacts;
- how a return value should be materialized, named, stored, and content-hashed;
- which parameters are durable public provenance and which are sensitive or
  incidental runtime state;
- what a meaningful Definition, contract, or terminal diagnostic is.

The NBA dogfood application makes those choices in small adapters beside its
domain boundaries: `season/oclp.py` for processing a game,
`modeling/oclp.py` for RAPM training, and `web_api/oclp_inference.py` for a
model-serving evaluation. Each adapter is ordinary explicit Python code.

## Declare a Definition beside its callable

`@oclp.definition` is deliberately limited to static Definition metadata. It
attaches a logical Definition ID, display name, and port contracts to a real
module-level callable, without wrapping the function or changing how it is
called. At publication time, `definition_record` adds the source actually used
for the run and derives the Python locator from the function itself.

```python
from oclp import GitSource, definition, definition_record
from oclp.models import PortDefinition


@definition(
    id="urn:example:definition:normalize-report",
    name="Normalize report",
    input_ports=(PortDefinition(name="source", media_types=("application/json",)),),
    output_ports=(PortDefinition(name="report", media_types=("application/json",)),),
)
def normalize_report(source: dict[str, object]) -> dict[str, object]:
    return {"title": str(source["title"]).strip()}


report_definition = definition_record(
    normalize_report,
    source=GitSource(
        repository="https://github.com/example/reports.git",
        commit="0123456789abcdef0123456789abcdef01234567",
        path="src/reports/normalize.py",
    ),
)

assert report_definition.implementation.locator == "reports.normalize.normalize_report"
```

The decorator prevents a hand-written locator from drifting away from its
implementation. It cannot determine the actual input or output Artifacts, a
meaningful run ID, or the test behind Evidence; those remain explicit
application responsibilities.

## The observation sequence

For a completed successful computation, the usual sequence is:

1. Materialize and hash the exact input bytes as one or more `Artifact`
   records (or an `ArtifactSet` for a named package).
2. Materialize and hash the output bytes the same way.
3. Publish a reusable `ComputationDefinition` with ports and implementation
   source information.
4. Publish one `Invocation` that binds the Definition, parameters, exact
   inputs, and exact outputs.
5. Publish ordered `LifecycleEvent` records for the observed attempt.
6. Publish `Evidence` for a contract you actually evaluated.

The record publication order is not the execution order. An observer may
publish after the computation finishes, while `occurred_at` and `observed_at`
preserve the actual chronology. This is how the current dogfood adapters leave
the application behavior unchanged.

```text
normal application work
    │
    ├── input bytes ──> Artifact(s) ─┐
    ├── function call                 ├──> Invocation ──> Events / Evidence
    └── output bytes ─> Artifact(s) ─┘
```

## A complete, explicit success observation

This example assumes the application already stores its immutable input and
output bytes at the shown locations. `DuckdbCatalog` stores the OCLP records;
it does not upload or retrieve those bytes. A production application may use a
file-backed publisher, an object store, or a registry instead.

```python
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from oclp import (
    Artifact,
    ComputationDefinition,
    Evidence,
    GitSource,
    Invocation,
    LifecycleEvent,
)
from oclp.catalog.duckdb import DuckdbCatalog
from oclp.models import ContractReference, Digest, Implementation, PortDefinition


def artifact_for_bytes(
    *,
    artifact_id: str,
    name: str,
    content: bytes,
    location: str,
    created_at: datetime,
) -> Artifact:
    return Artifact(
        id=artifact_id,  # a logical name, never urn:sha256:<content digest>
        name=name,
        media_type="application/json",
        digest=Digest(value=sha256(content).hexdigest()),
        size=len(content),
        created_at=created_at,
        locations=(location,),
    )


def record_daily_report(
    *,
    run_id: str,
    source_bytes: bytes,
    report_bytes: bytes,
    started_at: datetime,
    finished_at: datetime,
) -> None:
    input_artifact = artifact_for_bytes(
        artifact_id=f"urn:example:artifact:daily-source:{run_id}",
        name="Daily source",
        content=source_bytes,
        location=f"s3://example-reports/source/{run_id}.json",
        created_at=started_at,
    )
    output_artifact = artifact_for_bytes(
        artifact_id=f"urn:example:artifact:daily-report:{run_id}",
        name="Daily report",
        content=report_bytes,
        location=f"s3://example-reports/report/{run_id}.json",
        created_at=finished_at,
    )
    definition = ComputationDefinition(
        id="urn:example:definition:build-daily-report",
        name="Build daily report",
        implementation=Implementation(
            kind="python-callable",
            locator="reports.daily.build_report",
            # Substitute the actual repository and revision that supplied the
            # callable. A local worktree is recorded on the attempt Event.
            source=GitSource(
                repository="https://github.com/example/reports.git",
                commit="0123456789abcdef0123456789abcdef01234567",
                path="src/reports/daily.py",
            ),
        ),
        input_ports=(
            PortDefinition(name="source", media_types=("application/json",)),
        ),
        output_ports=(
            PortDefinition(name="report", media_types=("application/json",)),
        ),
    )

    with DuckdbCatalog(Path(".oclp/catalog.duckdb")) as catalog:
        definition_ref = catalog.publish(definition)
        input_ref = catalog.publish(input_artifact)
        output_ref = catalog.publish(output_artifact)
        invocation = Invocation(
            id=f"urn:example:invocation:build-daily-report:{run_id}",
            name=f"Build daily report – {run_id}",
            definition=definition_ref,
            parameters={"run_id": run_id},
            inputs={"source": (input_ref,)},
            outputs={"report": (output_ref,)},
            requested_outputs=("report",),
        )
        invocation_ref = catalog.publish(invocation)

        catalog.publish(
            LifecycleEvent(
                id=f"urn:example:event:daily-report-requested:{run_id}",
                invocation=invocation_ref,
                event_type="invocation-requested",
                occurred_at=started_at,
                sequence=0,
            )
        )
        catalog.publish(
            LifecycleEvent(
                id=f"urn:example:event:daily-report-started:{run_id}",
                invocation=invocation_ref,
                event_type="attempt-started",
                occurred_at=started_at,
                sequence=1,
                attempt_id=run_id,
            )
        )
        catalog.publish(
            LifecycleEvent(
                id=f"urn:example:event:daily-report-artifacts:{run_id}",
                invocation=invocation_ref,
                event_type="artifacts-published",
                occurred_at=finished_at,
                sequence=2,
                attempt_id=run_id,
                data={"outputs": {"report": output_ref.model_dump(mode="json")}},
            )
        )
        evidence_ref = catalog.publish(
            Evidence(
                id=f"urn:example:evidence:daily-report-contract:{run_id}",
                subject=invocation_ref,
                contract=ContractReference(
                    id="urn:example:contract:daily-report",
                    version="1",
                ),
                outcome="pass",
                observed_at=finished_at,
                details={"checks": [{"id": "report-is-valid-json", "outcome": "pass"}]},
            )
        )
        catalog.publish(
            LifecycleEvent(
                id=f"urn:example:event:daily-report-evidence:{run_id}",
                invocation=invocation_ref,
                event_type="evidence-published",
                occurred_at=finished_at,
                sequence=3,
                attempt_id=run_id,
                data={"evidence": evidence_ref.model_dump(mode="json")},
            )
        )
        catalog.publish(
            LifecycleEvent(
                id=f"urn:example:event:daily-report-terminal:{run_id}",
                invocation=invocation_ref,
                event_type="invocation-terminal",
                occurred_at=finished_at,
                sequence=4,
                attempt_id=run_id,
                status="succeeded",
            )
        )
```

The normal application owns `source_bytes`, `report_bytes`, their external
storage, and the semantic test behind the Evidence. The OCLP portion binds the
durable facts: which exact Definition, input, output, run parameters, and
attempt chronology were observed.

## Record a failure without inventing outputs

When the computation fails, publish the Definition, materialized inputs, and
an Invocation with `outputs=None`. Then publish requested and started Events,
followed by a terminal Event with `status="failed"` and a compact
`Diagnostic`. Do not create a placeholder output Artifact and do not claim an
`artifacts-published` Event.

```python
from oclp import Diagnostic, LifecycleEvent

catalog.publish(
    LifecycleEvent(
        id=f"urn:example:event:daily-report-failed:{run_id}",
        invocation=invocation_ref,
        event_type="invocation-terminal",
        occurred_at=finished_at,
        sequence=2,
        attempt_id=run_id,
        status="failed",
        diagnostic=Diagnostic(
            code="reports:daily:ValidationError",
            message="source did not contain a report date",
            stage="validate-source",
        ),
    )
)
```

If the failure happens before an input can be bound, record that honest fact:
use an empty `inputs` map and a diagnostic stage such as `fetch` or `adapt`.
Do not infer an input Artifact that the application did not successfully
identify.

## What to extract before adding a decorator

The repeated mechanics in the example are real candidates for an SDK helper:
publishing records, sequencing standard attempt Events, and translating an
exception to a terminal Event. The application-specific decisions should stay
visible.

A useful future API would therefore be an explicit session or adapter first:

```python
with observe_attempt(store, definition=report_definition, run_id=run_id) as attempt:
    attempt.bind_input("source", source_artifact)
    report = build_report(source)
    attempt.bind_output("report", materialize(report))
    attempt.check(report_contract)
```

A decorator could be thin sugar over that session, but only after the caller
supplies explicit functions for input binding, output materialization, ID
policy, and contracts. A bare `@oclp.observe` that hashes Python arguments or
return values would be convenient for demos but unreliable for files,
databases, services, generators, distributed work, and sensitive parameters.

Until that helper exists, prefer a small domain adapter at the computation
boundary. It is easy to test, does not change the job's behavior, and makes
the provenance decisions reviewable.

## Validate the result

At a useful boundary—usually in a test or a publication step—validate the
whole resolved graph:

```python
from oclp import validate_derivation_graph, validate_invocation_hierarchy

records = [definition, input_artifact, output_artifact, invocation]
validate_derivation_graph(records)
validate_invocation_hierarchy(records)
```

Pass all resolved records available to the validator. The derivation validator
checks Artifact → Invocation → Artifact bindings for cycles; the hierarchy
validator separately checks `parent_invocation` orchestration cycles.
