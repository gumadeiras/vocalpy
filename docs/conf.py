import os
import sys

sys.path.insert(0, os.path.abspath(os.path.pardir))

from vocalpy import __version__

project = "VocalPy"
copyright = "2020, Dietrich Lab"
author = "Gustavo Madeira Santana"
release = __version__
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
