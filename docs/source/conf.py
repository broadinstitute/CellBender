# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
from cellbender.base_cli import get_version


dir_, _ = os.path.split(__file__)
root_dir = os.path.abspath(os.path.join(dir_, '..', '..'))
sys.path.insert(0, root_dir)

# -- Project information -----------------------------------------------------

project = 'CellBender'
copyright = '2019, Data Sciences Platform (DSP), Broad Institute'
author = 'Stephen Fleming, Mehrtash Babadi'

# The full version, including alpha/beta/rc tags
version = get_version()
release = get_version()

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinxarg.ext',
    'sphinx_autodoc_typehints',
    'sphinxcontrib.programoutput',
    'sphinx.ext.intersphinx'
]

master_doc = 'index'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_favicon = '_static/design/favicon.ico'
html_css_files = [
    'theme_overrides.css',  # override wide tables in RTD theme
]
