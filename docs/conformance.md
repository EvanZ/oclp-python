# Conformance

The OCLP standard repository owns the Core specification, JSON Schemas,
canonical fixtures, and expected record digests. Optional profile contracts are
published independently in `oclp-profiles`. The Python SDK tests itself against
both external corpora instead of generating either contract from Python models.

## Run the SDK against the standard

Check out the standard alongside this repository, then point the SDK test at
it:

```bash
git clone https://github.com/EvanZ/open-computation-lifecycle.git ../open-computation-lifecycle
git clone https://github.com/EvanZ/oclp-profiles.git ../oclp-profiles
OCLP_STANDARD_ROOT=../open-computation-lifecycle \
OCLP_PROFILES_ROOT=../oclp-profiles \
  uv run pytest
```

The test parses valid records, rejects invalid records, checks RFC 8785
canonical JSON, verifies expected SHA-256 record digests, and validates the
standard's derivation fixtures. For each profile supported by this SDK, it also
validates the profile's independent schemas, vectors, canonical JSON, and
digest values.

## Cross-language contract

The Core standard and the profile repository each carry independent TypeScript
verifiers. A Core or profile change is not complete until its normative
specification, schemas, fixtures, expected canonical values, and independent
verifiers agree.

See [cross-language conformance](https://evanz.github.io/open-computation-lifecycle/protocol/conformance/)
for Core, and [OCLP Profiles](https://evanz.github.io/oclp-profiles/conformance/)
for optional profile contracts.
