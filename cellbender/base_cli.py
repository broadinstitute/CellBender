"""Command-line tool functionality.

Parses arguments and determines which tool should be called.

"""

import sys
import argparse
from abc import ABC, abstractmethod
from typing import Dict
import importlib


# New tools should be added to this list.
TOOL_NAME_LIST = ['remove-background']


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
    def validate_args(self, parser: argparse):
        """Do tool-specific argument validation, returning args."""
        pass

    @abstractmethod
    def run(self, args):
        """Run the tool using the parsed arguments."""
        pass


def generate_cli_dictionary() -> Dict[str, AbstractCLI]:
    # Add the tool-specific arguments using sub-parsers.
    cli_dict = dict(keys=TOOL_NAME_LIST)
    for tool_name in TOOL_NAME_LIST:
        # Note: tool name contains a dash, while folder name uses an underscore.
        # Generate the name of the module that contains the tool.
        module_cli_str_list = ["cellbender", tool_name.replace("-", "_"), "cli"]

        # Import the module.
        module_cli = importlib.import_module('.'.join(module_cli_str_list))

        # Note: the module must have a file named cli.py in the main
        # directory, containing a class named CLI, which implements AbstractCLI.
        cli_dict[tool_name] = module_cli.CLI()

    return cli_dict


def get_populated_argparser() -> argparse.ArgumentParser:
    # Set up argument parser.
    parser = argparse.ArgumentParser(
        prog="cellbender",
        description="CellBender is a software package for eliminating technical artifacts from high-throughput "
                    "single-cell RNA sequencing (scRNA-seq) data.")

    # Declare the existence of sub-parsers.
    subparsers = parser.add_subparsers(
        title="sub-commands",
        description="valid cellbender commands",
        dest="tool")

    for tool_name in TOOL_NAME_LIST:
        module_argparse_str_list = ["cellbender", tool_name.replace("-", "_"), "argparse"]
        module_argparse = importlib.import_module('.'.join(module_argparse_str_list))
        subparsers = module_argparse.add_subparser_args(subparsers)

    return parser


def main():
    """Parse command-line arguments and run specified tool.

    Note: Does not take explicit input arguments, but uses sys.argv inputs
    from the command line.

    """

    parser = get_populated_argparser()
    cli_dict = generate_cli_dictionary()

    # Parse arguments.
    if len(sys.argv) > 1:
        args = parser.parse_args(sys.argv[1:])

        # Validate arguments.
        args = cli_dict[args.tool].validate_args(args)

        # Run the tool.
        cli_dict[args.tool].run(args)

    else:

        parser.print_help()
