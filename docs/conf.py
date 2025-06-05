# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import subprocess

# -- Read the Doc configuration ----------------------------------------------


def configureDoxyfile(input_dir, output_dir):
    with open("Doxyfile.in") as file:
        filedata = file.read()

    filedata = filedata.replace("@PROJECT_SOURCE_DIR@", input_dir)
    filedata = filedata.replace("@DOXYGEN_OUTPUT_DIR@", output_dir)
    filedata = filedata.replace("@DOXYGEN_DOT_FOUND@", "NO")
    filedata = filedata.replace("@DOXYGEN_TAGFILE@", "")

    with open("Doxyfile", "w") as file:
        file.write(filedata)


# Check if we're running on Read the Docs' servers
read_the_docs_build = os.environ.get("READTHEDOCS", None) == "True"

breathe_projects = {}

if read_the_docs_build:
    input_dir = "../"
    output_dir = "_build/"
    configureDoxyfile(input_dir, output_dir)
    subprocess.call("doxygen", shell=True)
    breathe_projects["IgANet"] = output_dir + "xml"


# -- Project information -----------------------------------------------------

project = "IgANet"
copyright = "2021-2025, Matthias Möller (m.moller@tudelft.nl)"
author = "Matthias Möller"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx.ext.mathjax", "breathe", "sphinxcontrib.tikz"]

# Tikz configuration
tikz_resolution = 250
tikz_proc_suite = "GhostScript"
tikz_tikzlibraries = "quantikz"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

highlight_language = "c++"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "style_nav_header_background": "#00a6d6",
    "logo_only": False,
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_logo = "_static/IgANet_logo_white.png"

# Breathe Configuration
breathe_default_project = "IgANet"
