from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("cellbender")
except PackageNotFoundError:
    __version__ = "unknown"
