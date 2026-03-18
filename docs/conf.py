# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import datetime

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'skyloc'
copyright = f'2024-{datetime.now().year}, ysBach'
author = 'ysBach'

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx_copybutton",
    "numpydoc",
]

# Napoleon settings for NumPy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = False
napoleon_use_rtype = True
napoleon_preprocess_types = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": True,
    "ignore-module-all": True,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Mock imports for modules that might not be available during doc builds
autodoc_mock_imports = [
    "kete",
    "plotly",
    "spiceypy",
    "numba",
    "healpy",
    "pqfilt",
]



# Autosummary
autosummary_generate = True

# Default role for backticks (allows `SPHEREx` instead of :obj:`SPHEREx`)
default_role = "obj"

# -- Intersphinx configuration -----------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "astroquery": ("https://astroquery.readthedocs.io/en/latest/", None),
}

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
}

# Suppress warnings for missing references in external packages
nitpicky = False

# -- Templates ---------------------------------------------------------------
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Source suffix -----------------------------------------------------------
source_suffix = ".rst"
master_doc = "index"

