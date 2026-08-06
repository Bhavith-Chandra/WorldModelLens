# Configuration file for the Sphinx documentation builder.

from __future__ import annotations

import os
import sys

ROOT = os.path.abspath("..")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

autodoc_mock_imports = [
    "torch",
    "numpy",
    "gymnasium",
    "einops",
    "plotly",
    "scipy",
    "sklearn",
    "pandas",
    "tqdm",
    "rich",
    "yaml",
    "typer",
    "huggingface_hub",
]

project = "World Model Lens"
author = "World Model Lens Team"
copyright = "2023, World Model Lens Team"
release = "0.2.0"

master_doc = "index"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Bhavith-Chandra/WorldModelLens",
            "icon": "fa-brands fa-github",
        }
    ],
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "footer_start": ["copyright"],
    "footer_end": [],
}

html_context = {
    "github_user": "Bhavith-Chandra",
    "github_repo": "WorldModelLens",
    "github_version": "master",
    "doc_path": "docs",
}

myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}
