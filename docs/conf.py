import os
import sys
import tomllib

from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.pardir))

ROOT = Path(__file__).resolve().parent.parent
with (ROOT / "pyproject.toml").open("rb") as pyproject_file:
    PYPROJECT = tomllib.load(pyproject_file)

project = "VocalPy"
copyright = "2020, Dietrich Lab"
author = "Gustavo Madeira Santana"
release = PYPROJECT["project"]["version"]
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
]
source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
