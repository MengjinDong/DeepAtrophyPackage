# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'DeepAtrophy'
copyright = '2025, Mengjin Dong'
author = 'Mengjin Dong'
release = '0.0.3'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# extensions = []
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', "myst_parser", 'sphinx.ext.todo',]
templates_path = ['_templates']
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
# html_theme = 'sphinx_rtd_theme' # standard theme
html_theme = 'furo'
html_static_path = ['_static']

# html_theme_options = { # for sphinx_rtd_theme
#     'collapse_navigation': False,
#     'sticky_navigation': True,
#     'navigation_depth': 4,
#     'includehidden': True,
#     'titles_only': False
# }

html_theme_options = {
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
}

# Output file base name for HTML help builder.
htmlhelp_basename = 'deepatrophy'


# -- Options for autodoc -----------------------------------------------------
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

