# Artifact sets

An ArtifactSet is one immutable, named collection of exact Artifact references.
Use it when a consumer needs a package—such as a model, its feature contract,
and its evaluation report—as one lineage input or output.

## Create a set from one Computation

Use `ComputationArtifactSet` when one Computation emits all members:

```python
from oclp import ComputationArtifactSet, JsonArtifact, computation


@computation(
    name="Evaluate candidate",
    outputs={
        "model": JsonArtifact(name="Candidate model"),
        "evaluation": JsonArtifact(name="Candidate evaluation"),
    },
    artifact_set=ComputationArtifactSet(
        name="Evaluated candidate",
        port="candidate_release",
        members={
            "model": ("model", "model"),
            "evaluation": ("evaluation", "validation-report"),
        },
    ),
)
def evaluate_candidate() -> dict[str, object]:
    return {"model": {"version": 1}, "evaluation": {"rmse": 212.4}}
```

`candidate_release` becomes another real output of the same Execution. It does
not create a packaging Computation or a synthetic Execution.

## Assemble a release across Computations

For members produced by separate Computations, declare each contribution with
`@artifact_set` outside its `@computation`. Matching names are assembled when
the observed run completes successfully:

```python
from oclp import artifact_set, computation


@artifact_set(
    name="Demand model release",
    output_port="model",
    role="model",
)
@computation(...)
def train_model(...): ...


@artifact_set(
    name="Demand model release",
    output_port="evaluation",
    role="validation-report",
)
@computation(...)
def evaluate_model(...): ...
```

Each member must materialize exactly once. The SDK does not guess across
workers or invent a selection policy when a declared member is ambiguous.

## Assemble a terminal collection across runs

Use `assemble_artifact_set(...)` when a scheduler needs a visible terminal
node that collects exact upstream Artifact handles from separate runs. It
declares only the direct ArtifactSet operation: it does not claim to be an
Artifact or fabricate an Execution. See the [Dagster integration](dagster.md)
for this canonical declaration in a cross-run asset graph.

```python
from oclp import ArtifactHandle, assemble_artifact_set


@assemble_artifact_set(
    name="Demand model release",
    members={
        "model": ("model", "model"),
        "evaluation": ("evaluation", "validation-report"),
    },
)
def assemble_release(
    model: ArtifactHandle,
    evaluation: ArtifactHandle,
) -> None:
    pass
```

Call `assemble_release(model, evaluation)` inside an observed run. The return
value is the resulting `ArtifactSetHandle`; the marker body may remain empty.

When membership itself is genuinely dynamic, call
`observed.publish_artifact_set(...)` with exact handles instead. It is the
imperative direct-collection API; it likewise creates no synthetic
Computation or Execution.

Use `ArtifactSetHandle.member(name)` or `load_member(name, target_type)` to
retrieve a member. A Computation that consumes the package should declare
`artifact_set_input(...)`, preserving the ArtifactSet reference as one real
Execution input.
