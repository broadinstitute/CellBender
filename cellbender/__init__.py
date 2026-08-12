from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cellbender")
except PackageNotFoundError:
    __version__ = "unknown"

try:
    version("jupyter_contrib_nbextensions")
    import warnings

    warnings.warn(
        "\n\n"
        "WARNING: 'jupyter_contrib_nbextensions' is installed in your environment.\n"
        "This package is incompatible with notebook>=7 and will cause CellBender's\n"
        "report generation to fail with:\n"
        "    ModuleNotFoundError: No module named 'notebook.services'\n"
        "Please uninstall it to fix this:\n"
        "    pip uninstall jupyter_contrib_nbextensions\n"
        "We recommend stopping this run and uninstalling jupyter_contrib_nbextensions before proceeding\n",
        stacklevel=2,
    )
except PackageNotFoundError:
    pass
