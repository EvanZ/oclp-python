# OCLP Python SDK

The reference Python implementation of the [Open Computation Lifecycle
Protocol](https://github.com/EvanZ/open-computation-lifecycle).

OCLP is a language-neutral standard for durable computation provenance. This
package provides Python models, canonical JSON serialization, digest helpers,
validation, portable profile helpers, and an optional local DuckDB catalog.

The protocol specification, schemas, examples, and cross-language conformance
vectors live in the separate
[`open-computation-lifecycle`](https://github.com/EvanZ/open-computation-lifecycle)
repository. The SDK follows that specification; it does not define it.

Full SDK documentation is published at
[evanz.github.io/oclp-python](https://evanz.github.io/oclp-python/).

## Install

```bash
pip install "oclp[duckdb] @ git+https://github.com/EvanZ/oclp-python.git@main"
```

For a reproducible deployment, replace `main` with a reviewed immutable commit
SHA. The optional `duckdb` extra installs the local record catalog.

For development:

```bash
git clone https://github.com/EvanZ/oclp-python.git
cd oclp-python
uv sync --group dev
uv run pytest
uv run ruff check .
```

## Scope

`oclp` is deliberately an SDK, not an orchestrator or a hosted provenance
service. Applications decide when to create records; consumers such as
[Cyclops](https://github.com/EvanZ/oclp-explorer) can then inspect them.

The `duckdb` extra supplies a simple local catalog for resolving
content-bound records and their locations. It is optional implementation
infrastructure, not a requirement of the OCLP standard.

## End-to-end example

The [bike-demand service demo](examples/bike-demand-service/README.md) is a
staged, self-contained consumer project for returning to an end-to-end OCLP
design: source data, feature preparation, temporal model folds, evaluation,
model release packaging, and FastAPI inference. It has its own dependencies so
the SDK's core installation remains small.
