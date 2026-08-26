# Conformance

The OCLP standard repository owns the normative text, JSON Schemas, canonical
fixtures, and expected record digests. The Python SDK tests itself against that
external corpus instead of generating the contract from Python models.

## Run the SDK against the standard

Check out the standard alongside this repository, then point the SDK test at
it:

```bash
git clone https://github.com/EvanZ/open-computation-lifecycle.git ../open-computation-lifecycle
OCLP_STANDARD_ROOT=../open-computation-lifecycle \
  uv run pytest tests/test_standard_conformance.py
```

The test parses valid records, rejects invalid records, checks RFC 8785
canonical JSON, verifies expected SHA-256 record digests, and validates the
standard's derivation fixtures.

## Cross-language contract

The standard repository also carries an independent TypeScript verifier. A
protocol change is not complete until the normative specification, schemas,
fixtures, expected canonical values, and independent verifiers agree.

See [cross-language conformance](https://evanz.github.io/open-computation-lifecycle/protocol/conformance/)
for the contract and change process.
