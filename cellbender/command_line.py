"""Command-line tool functionality.

Parses arguments and determines which tool should be called.

"""

import sys
import argparse
from abc import ABC, abstractmethod
import importlib


# New tools should be added to this list.
TOOL_LIST = ['remove-background']


class AbstractCLI(ABC):
    """Abstract class for cellbender command-line interface tools.

    Note:
        Tools are called from the command line using
        $ cellbender TOOL-NAME --optional_arg1 optional_val1 ...

    """

    @abstractmethod
    def get_name(self) -> str:
        """Return the command-line name of the tool."""
        pass

    @abstractmethod
    def add_subparser_args(self, parser: argparse) -> argparse:
        """Add tool-specific arguments, returning a parser."""
        pass

    @abstractmethod
    def validate_args(self, parser: argparse):
        """Do tool-specific argument validation, returning args."""
        pass

    @abstractmethod
    def run(self, args):
        """Run the tool using the parsed arguments."""
        pass


def main():
    """Parse command-line arguments and run specified tool.

    Note: Does not take explicit input arguments, but uses sys.argv inputs
    from the command line.

    """

    # Set up argument parser.
    parser = argparse.ArgumentParser(prog="cellbender",
                                     description="CellBender command-line "
                                                 "tools for scRNA-seq "
                                                 "data analysis.")
    parser.add_argument("cellbender", nargs=1, type=str,
                        help="'cellbender', the name of the package")

    # Declare the existence of sub-parsers.
    subparsers = parser.add_subparsers(title="sub-commands",
                                       description="valid cellbender commands",
                                       dest="tool")

    # Add the tool-specific arguments using sub-parsers.
    cli = dict(keys=TOOL_LIST)
    for tool in TOOL_LIST:

        # Note: tool name contains a dash, while folder name uses an underscore.

        # Generate the name of the module that contains the tool.
        module_str_list = ["cellbender", tool.replace("-", "_"), "command_line"]

        # Import the module.
        module = importlib.import_module('.'.join(module_str_list))

        # Note: the module must have a file named command_line.py in the main
        # directory, containing a class named CLI, which implements AbstractCLI.
        cli[tool] = module.CLI()
        subparsers = cli[tool].add_subparser_args(subparsers)

    # Parse arguments.
    args = parser.parse_args(sys.argv)

    if args.tool is not None:

        # Validate arguments.
        args = cli[args.tool].validate_args(args)

        # Run the tool.
        cli[args.tool].run(args)
