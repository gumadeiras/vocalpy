import os
import sys
import tomllib

from pathlib import Path

from sphinx.application import Sphinx

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
    "sphinx_sitemap",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]
autodoc_mock_imports = [
    "cv2",
    "joblib",
    "matplotlib",
    "numpy",
    "pandas",
    "PIL",
    "scipy",
    "seaborn",
    "skimage",
    "soundfile",
    "torch",
    "torchvision",
    "yaml",
]
source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "furo"
html_static_path = ["_static"]
html_baseurl = "https://vocalpy.gumadeiras.com/"
html_extra_path = ["CNAME", ".nojekyll", "robots.txt"]
sitemap_url_scheme = "{link}"
sitemap_excludes = [
    "index.html",
    "genindex.html",
    "py-modindex.html",
    "search.html",
]
html_theme_options = {
    "source_repository": "https://github.com/gumadeiras/vocalpy/",
    "source_branch": "main",
    "source_directory": "docs/",
}


def _set_homepage_canonical(
    app: Sphinx,
    pagename: str,
    _templatename: str,
    context: dict[str, object],
    _doctree: object,
) -> None:
    """Use the public root URL for the documentation home page."""
    if pagename == "index":
        context["pageurl"] = app.config.html_baseurl


def setup(app: Sphinx) -> None:
    """Register the home-page canonical URL correction.

    Args:
        app: Current Sphinx application.

    Returns:
        None.
    """
    app.connect("html-page-context", _set_homepage_canonical, priority=900)
