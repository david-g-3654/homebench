# Releasing homebench to PyPI

Published to PyPI as **`homebench`** (the name `localbench` was already taken,
which is why the whole project uses `homebench` — distribution, command, and
`import homebench`). Uses the **setuptools** build backend (`pyproject.toml`),
built with [`build`](https://pypi.org/project/build/) and uploaded with
[`twine`](https://pypi.org/project/twine/).

## One-time setup

- Create accounts on [TestPyPI](https://test.pypi.org/) and [PyPI](https://pypi.org/).
- Create an **API token** on each (Account settings → API tokens). twine reads it
  from `~/.pypirc` or the `TWINE_USERNAME`/`TWINE_PASSWORD` env vars
  (`TWINE_USERNAME=__token__`, `TWINE_PASSWORD=pypi-...`).
- Install the tooling:

  ```bash
  pip install -e ".[publish]"     # installs build + twine
  ```

## Cut a release

1. Bump `version` in `pyproject.toml` (and tag it in git).
2. Build a clean sdist + wheel:

   ```bash
   rm -rf dist build *.egg-info
   python -m build
   ```

   This produces `dist/homebench-<version>.tar.gz` (sdist) and
   `dist/homebench-<version>-py3-none-any.whl` (wheel).

3. Validate the metadata renders correctly for PyPI:

   ```bash
   twine check dist/*
   ```

4. Upload to **TestPyPI** first and smoke-test the install:

   ```bash
   twine upload --repository testpypi dist/*
   pipx install --index-url https://test.pypi.org/simple/ \
       --pip-args="--extra-index-url https://pypi.org/simple" homebench
   homebench --version
   ```

5. Upload to **PyPI**:

   ```bash
   twine upload dist/*
   ```

6. Tag and push:

   ```bash
   git tag v<version> && git push --tags
   ```

## Notes

- The project name is `homebench` throughout — distribution, the `homebench`
  command (`[project.scripts]`), and `import homebench` (`[project].name`).
- Version numbers can't be reused on PyPI — always bump `version` in
  `pyproject.toml` before re-uploading (current: 0.2.0).
- Optional extras: `homebench[yaml]` (YAML task packs), `homebench[dev]`
  (tests), `homebench[publish]` (build/twine).
