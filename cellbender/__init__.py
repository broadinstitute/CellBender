from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cellbender")
except PackageNotFoundError:
    __version__ = "unknown"
