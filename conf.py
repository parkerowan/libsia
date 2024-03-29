# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('..'))
# from recommonmark.transform import AutoStructify

import re

# Get the project version
def get_version() -> str:
    with open("version.txt") as f:
        content = f.readlines()
    vdict = {}
    for x in content:
        pattern = "(\S*) (\d[0-9]*)"
        match = re.search(pattern, x.strip())
        vdict[match.group(1)] = match.group(2)
    ver = "{}.{}.{}".format(vdict["VERSION_MAJOR"], vdict["VERSION_MINOR"],
                            vdict["VERSION_PATCH"])
    return ver

# -- Project information -----------------------------------------------------

project = 'libSIA'
copyright = '2018-2023, Parker Owan'
author = 'Parker Owan'

# The full version, including alpha/beta/rc tags
release = get_version()

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'myst_parser',
    'nbsphinx',
    'sphinx_markdown_tables',
    'sphinx.ext.mathjax',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['docs/_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['build', 'docs/_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['docs/_static']

# Link URLs
html_theme_options = {
    "github_url": "https://github.com/parkerowan/libsia",
    "gitlab_url": "https://gitlab.com/parkerowan/libsia",
}

# Source files
source_suffix = ['.rst', '.md']
