- External GitHub links on the docs landing page are now the canonical path to
  installation and notebook curation guidance.
  Why it matters: offline docs readers will not see that material in the built
  site itself.
  Smallest documentation fix: add dedicated Sphinx pages later if offline docs
  become important.
  Follow-up code fix: none.

- `docs/conf.py` now imports `vocalpy.__version__`.
  Why it matters: if `vocalpy/__init__.py` ever grows import-heavy side effects,
  docs build could become more fragile.
  Smallest documentation fix: keep `vocalpy.__init__` lightweight by convention.
  Follow-up code fix: if that changes, read version from `pyproject.toml`
  without importing the package.
