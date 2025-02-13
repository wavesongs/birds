# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'WaveSongs'
copyright = '2025, Sebastian Aguilera Novoa'
author = 'Sebastian Aguilera Novoa'
release = '0.0.3b1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",
    # "myst_parser",
    "sphinx_design",
    'sphinx.ext.napoleon',
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx_copybutton",
    # 'sphinxcontrib.bibtex',
    "IPython.sphinxext.ipython_console_highlighting",
]

# bibtex_bibfiles = ['references.bib']

# MyST-NB settings
nb_execution_mode = "auto"
nb_execution_timeout = 90

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]
myst_url_schemes = ("http", "https", "mailto")

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "_templates"]

coverage_show_missing_items = True
autosummary_generate = True  # Turn on sphinx.ext.autosummary
templates_path = ["_templates"]
# autodoc_member_order = "bysource"

# Strip input prompts from copied code
# copybutton_prompt_text = ">>> "
# copybutton_prompt_text = (
#     r">>> |^\d+|\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
# )
copybutton_prompt_is_regexp = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']


html_theme_options = {
    "repository_url": "https://github.com/wavesongs/wavesongs",
    "repository_branch": "main",
    "path_to_docs": "docs/",
    "use_repository_button": True,
    "use_download_button": True,
    "use_fullscreen_button": True,
    "home_page_in_toc": True,
    "use_issues_button": True,
    "launch_buttons": {
        "colab_url": "https://colab.research.google.com"
    },
    "icon_links": [
        {
            # Label for this link
            "name": "GitHub",
            # URL where the link will redirect
            "url": "https://github.com/wavesongs/wavesongs",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fa-brands fa-square-github",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
        }
   ]
    # "logo_only": True,
    # "extra_navbar": False,
}

# html_logo = "path/to/myimage.png"
html_title = "WaveSongs"


autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    # "special-members": "__init__",
    # "member-order": "bysource",
}