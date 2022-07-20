# -- Project information -----------------------------------------------------
project = "FullSubNet"
author = "HAO Xiang <haoxiangsnr@gmail.com>"
project_copyright = "2022, HAO Xiang"

# -- MetaConfig configuration ---------------------------------------------------
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
extensions = [
    "myst_parser",  # markdown file parser.
    "sphinx.ext.todo",  # enable the todo.
    "sphinx.ext.autodoc",  # provide automatic documentation for module (*.py), class, function, and  typehints.
    "sphinx.ext.autosummary",  # auto-generate the summary (include links) of the modules.
    "sphinx.ext.intersphinx",  # enable cross-referencing between Sphinx projects.
    "sphinx.ext.viewcode",  # add a helpful link to the source code of each object in the API reference sheet.
    "sphinx.ext.mathjax",  # enable math support in the documentation.
    "sphinx.ext.napoleon",  # [ordered] parse our docstrings and generate Google-style docstrings.
    "sphinxcontrib.autodoc_pydantic",  # generate the suitable docstrings to pydantic models.
]

# -- Extension configuration -------------------------------------------------
napoleon_numpy_docstring = False
napoleon_attr_annotations = True
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
autosummary_generate = True
autodoc_mock_imports = ["soundfile", "gpuRIR"]
autodoc_pydantic_model_signature_prefix = "Config"
autodoc_pydantic_member_order = "bysource"
autodoc_pydantic_model_show_field_summary = False
autodoc_pydantic_model_show_json = False
autodoc_pydantic_model_show_validator_members = False
autodoc_pydantic_model_show_validator_summary = False
autodoc_pydantic_model_summary_list_order = "bysource"
autodoc_pydantic_model_list_validators = False
autodoc_pydantic_field_signature_prefix = "option"

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_context = {
    "display_github": True,  # edit on Github, see https://github.com/readthedocs/sphinx_rtd_theme/issues/529
    "github_user": "haoxiangsnr",
    "github_repo": "FullSubNet",
    "github_version": "main",
}
html_static_path = ["_static"]
html_css_files = [
    "css/custom.css",
]
