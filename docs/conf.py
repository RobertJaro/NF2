import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True

os.environ.setdefault("LC_ALL", "C")
os.environ.setdefault("LANG", "C")
os.environ.setdefault("SUNPY_CONFIGDIR", "/tmp/sunpy")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("."))

from generate_reference import generate

generate(Path(__file__).parent)

project = "NF2"
author = "Robert Jarolim"
release = "0.4.1"

extensions = [
    "myst_parser",
    "sphinxext.opengraph",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_logo = "_static/logo.jpg"
html_favicon = "_static/logo.jpg"
html_theme_options = {
    "github_url": "https://github.com/RobertJaro/NF2",
    "show_toc_level": 2,
}

ogp_site_url = "https://nf2.readthedocs.io/en/latest/"
ogp_site_name = "NF2"
ogp_image = "https://raw.githubusercontent.com/RobertJaro/NF2/main/images/logo.jpg"
ogp_type = "website"

autodoc_member_order = "bysource"
autodoc_typehints = "description"
myst_heading_anchors = 3
