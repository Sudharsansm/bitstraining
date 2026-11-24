# Releasing Bitstraining to PyPI (from GitHub, no Docker)

Releases are built and published by `.github/workflows/release.yml` using PyPI **trusted publishing**
(OIDC). No API token is stored in GitHub. Docker is not used anywhere in this pipeline.

## One-time setup

1. Create the GitHub repository and push this project (default branch `main`).
2. Create accounts on https://pypi.org and (optional rehearsal) https://test.pypi.org.
3. On PyPI: *Your account > Publishing > Add a new pending publisher*
   - PyPI project name: `bitstraining`
   - Owner: your GitHub user/org; Repository: your repo name
   - Workflow filename: `release.yml`
   - Environment name: `pypi`
   (Repeat on TestPyPI with environment `testpypi` if you want rehearsals.)
4. On GitHub: *Settings > Environments*: create `pypi` (and `testpypi`). For `pypi`, add yourself as a
   required reviewer so every publish needs a manual approval click.
5. Add `[project.urls]` to `pyproject.toml` (Homepage, Repository, Issues) with your repo URL.

## Each release

1. Edit `__version__` in `bitstraining/__init__.py` (the only place the version lives).
2. Add a CHANGELOG entry. Say which earlier *results* could be affected, not just API changes.
3. Run locally: `pip install -e ".[dev]" build twine && pytest && ruff check bitstraining tests`
   then `python -m build && twine check dist/*`.
4. Commit, push, wait for CI (Linux, Windows, macOS) to pass.
5. Tag and publish a GitHub Release named `vX.Y.Z` (e.g. `v0.1.0`). Publishing the release starts the workflow:
   - checks the tag equals `__version__` (fails otherwise),
   - builds sdist + wheel and runs `twine check`,
   - installs the **wheel** on Linux/Windows/macOS, runs `bitstraining quickstart` and the test suite
     against it,
   - publishes to PyPI after the environment approval.
6. Verify: `pip install bitstraining==X.Y.Z` in a fresh virtualenv.

## Rehearsal

Actions tab > *release* > *Run workflow* builds, tests and uploads to TestPyPI only.
Install with `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple bitstraining`.

## Notes

* PyPI versions are immutable. A bad release is fixed by yanking it and releasing the next version.
* The name `bitstraining` was unclaimed when checked; the first publish through a pending publisher claims it.
