# Evidence

Evidence records a reusable, source-bound evaluation of a real Computation
output. It is not a metric type and it does not replace application control
flow: an evaluator reports whether one explicit rule passed, failed, or could
not be evaluated.

## Declare an evaluation rule

Use `@evidence` for a rule that accepts exactly one value and returns
`"pass"`, `"fail"`, or `"error"`:

```python
from oclp import JsonArtifact, computation, evidence


@evidence(name="Validation quality")
def validation_quality(metrics: dict[str, float]) -> str:
    return "pass" if metrics["rmse"] <= 250 else "fail"


@computation(
    name="Evaluate candidate",
    outputs={"metrics": JsonArtifact(name="Validation metrics")},
    requires=(validation_quality,),
)
def evaluate_candidate() -> dict[str, object]:
    return {"metrics": {"rmse": 212.4}}
```

The SDK publishes one Evidence record for the evaluator's selected output. A
returned `"fail"` or `"error"`, and an evaluator exception, produce a compact
Diagnostic. The original Execution and its outputs are still published.

## Required Evidence controls the Execution result

Every evaluator in `requires=` runs before the Execution's terminal Event. An
Execution can be `succeeded` only when every required evaluator has Evidence
with `outcome="pass"`; otherwise its terminal status is `failed`.

The surrounding run continues by default. To stop application control flow
after the failing Execution has been completely published, configure the
workflow with `required_evidence_policy="raise"`. See [Runs](runs.md) for that
run-level policy and [Computations and Executions](computations-and-executions.md)
for declared output ports and parameters.

## When to use Evidence

Use Evidence for a durable conclusion about a record: validation quality,
schema conformance, a promotion gate, or a safety check. Keep ordinary numeric
measurements in an output Artifact; an optional integration such as MLflow can
then mirror selected values for comparison without changing OCLP provenance.
