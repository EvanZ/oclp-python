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
pip install "oclp[duckdb]"
```

Omit `[duckdb]` if the local record catalog is not needed. The SDK is an
experimental pre-1.0 package: APIs and protocol support may change between
minor releases. Pin a published version for reproducible deployments.

For unreleased changes, install directly from GitHub:

```bash
pip install "oclp[duckdb] @ git+https://github.com/EvanZ/oclp-python.git@main"
```

Replace `main` with a reviewed immutable commit SHA when testing an unreleased
build reproducibly.

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

## Releases

GitHub `main` contains ongoing development. PyPI receives intentional,
versioned releases published from a GitHub Release through PyPI Trusted
Publishing. Early releases use PEP 440 pre-release versions (such as
`0.3.0a0`) while the OCLP protocol and SDK API are still evolving. See the
[GitHub releases](https://github.com/EvanZ/oclp-python/releases) for changes
and installation versions. Maintainers can follow the
[release instructions](docs/releasing.md).

## End-to-end example

The [bike-demand service demo](examples/bike-demand-service/README.md) is a
staged, self-contained consumer project for returning to an end-to-end OCLP
design: source data, feature preparation, temporal model folds, evaluation,
model release packaging, and FastAPI inference. It has its own dependencies so
the SDK's core installation remains small.
