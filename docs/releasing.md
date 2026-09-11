# Releasing the SDK

The SDK is developed on GitHub and published to PyPI only through intentional,
versioned GitHub Releases. Before the first release, configure a pending PyPI
trusted publisher with these values:

- PyPI project: `oclp`
- GitHub owner: `EvanZ`
- GitHub repository: `oclp-python`
- Workflow: `publish.yml`
- Environment: `pypi`

This reserves the package name and allows PyPI to accept OpenID Connect tokens
from the release workflow. It does not require a stored PyPI API token.

## Publish a release

1. Ensure `main` is green and decide the next PEP 440 version. While the SDK
   and protocol are pre-1.0, use an alpha or beta version such as `0.3.0a1`.
2. Change `[project].version` in `pyproject.toml`, update user-facing release
   notes, and merge the release-preparation commit to `main`.
3. Create an annotated tag named `v<version>` at that exact commit—for example,
   `v0.3.0a1`—and create a GitHub Release from the tag with its release notes.
4. Publish the GitHub Release. The `Publish to PyPI` workflow verifies that the
   tag matches the project version, builds the wheel and source distribution,
   and publishes both with PyPI Trusted Publishing.
5. Verify the release on PyPI, install the exact published version into a
   clean environment, and add the PyPI link to the GitHub Release if needed.

Never upload a build manually from a development machine. A release that needs
to be withdrawn should be yanked on PyPI and superseded by a new version; an
uploaded version cannot be replaced.
