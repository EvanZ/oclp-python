# Computations and Executions

A Computation declares reusable work: its implementation basis, interface,
parameter schema, outputs, and required Evidence. An Execution records one
actual invocation of that Computation with its exact inputs, effective
parameters, outputs, Evidence, and ordered Events.

The SDK keeps the declaration beside an ordinary Python function. It publishes
records only inside an active observed run; outside one, the callable retains
ordinary Python behavior.

## Declare a Computation

Use `@computation` to declare inputs, durable outputs, and required Evidence:

```python
from oclp import CsvArtifact, JsonArtifact, computation


@computation(
    name="Train one temporal fold",
    inputs={"features": CsvArtifact},
    outputs={"metrics": JsonArtifact(name="Fold metrics")},
)
def train_fold(
    features: object,
    *,
    fold_number: int,
    training_window: str = "pre-holdout",
) -> dict[str, object]:
    return {"metrics": {"fold_number": fold_number, "rmse": 212.4}}
```

The input mapping describes durable Artifact ports. Each input-port name must
match a callable parameter. Each output port has an explicit Artifact
representation and human-readable Artifact name.

Non-Artifact arguments become declared `ParameterDefinition` values on the
Computation. Required Python arguments stay required; JSON-compatible defaults
become schema defaults and are recorded as effective Execution parameters.
This avoids a second `parameters=` declaration that can drift from the
function's real call signature.

## Observe real invocations

An observed run supplies the source basis and publisher. The first invocation
of a decorated callable publishes one source-bound Computation record; every
call publishes its own Execution:

```python
from oclp import observe_run, run


@run(name="Daily training")
def train(features: object) -> None:
    train_fold(features, fold_number=1)
    train_fold(features, fold_number=2)


with observe_run(train, publisher=publisher, source=source) as observed:
    train(features)
```

This produces one Computation and two Executions. Each Execution records the
exact Artifact references supplied at its input ports and the effective values
of `fold_number` and `training_window`. Retries and reruns are distinct
Executions; no function call is collapsed by name or value equality.

## Materialize declared outputs

When a Computation declares an output, the runtime persists its returned value
through the corresponding Artifact representation and connects the resulting
Artifact reference to the Execution:

```python
result = train_fold(features, fold_number=1)
metrics = observed.outputs_for(result)["metrics"]
```

The wrapper returns the original Python value. `outputs_for(...)` returns exact
`ArtifactHandle` values for the materialized ports. Pass a handle to another
declared Computation when the downstream call should reload verified bytes; in
one process, passing the raw result directly also preserves the existing
materialized binding.

For several Artifacts of one type, declare `many(CsvArtifact)` on the input
port; the callable receives its normal ordered collection after the SDK
verifies and loads each member.

See [Artifacts](artifacts.md) for acquisition decorators, representations, and
handles. See [Artifact sets](artifact-sets.md) when outputs should become a
single package.

## Name records deliberately

Every record-producing declaration requires an application-supplied `name`.
Use `description` for longer human-facing text and `annotations` for structured
application metadata. The SDK never derives these labels from a function name,
timestamp, digest, or storage path.

The produced Core records receive opaque UUIDs. An Artifact payload's SHA-256
digest identifies bytes, while the Artifact record UUID identifies that one
immutable observation; publishing equal bytes twice therefore creates two
distinct Artifact records.

## Evidence and terminal status

Add `requires=(...)` to declare the Evidence gates that determine whether an
Execution succeeds. The runtime publishes outputs and all Evidence before it
emits the terminal Event. A required failure makes that Execution `failed`.

See [Evidence](evidence.md) for evaluator declarations and [Runs](runs.md) for
the optional policy that raises after a failed Execution has been published.
