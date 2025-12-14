"""Sphinx configuration for spark-bestfit documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

from spark_bestfit import __version__

project = "spark-bestfit"
copyright = "2025, Dustin Smith"
author = "Dustin Smith"
version = __version__
release = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}
