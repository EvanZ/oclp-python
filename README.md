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

## Install

```bash
pip install "oclp[duckdb]"
```

For development:

```bash
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
