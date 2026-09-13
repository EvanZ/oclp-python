# How to Use

## Install

Install the latest published experimental SDK release:

```bash
pip install "oclp[duckdb]"
```

Omit `[duckdb]` when the local catalog is not needed. The SDK is pre-1.0, so
pin a published version when a deployment needs reproducible behavior:

```bash
pip install "oclp[duckdb]==0.3.0a0"
```

To test unreleased changes, install from a reviewed immutable Git commit:

```bash
pip install "oclp[duckdb] @ git+https://github.com/EvanZ/oclp-python.git@<commit-sha>"
```

For SDK development:

```bash
git clone https://github.com/EvanZ/oclp-python.git
cd oclp-python
uv sync --all-groups
```
