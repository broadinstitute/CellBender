"""Command-line tool functionality for remove-background."""

import torch
from cellbender.remove_background.data.dataset import SingleCellRNACountsDataset
from cellbender.remove_background.train import run_inference
from cellbender.base_cli import AbstractCLI
from cellbender.remove_background import consts

import logging
import os
import sys
from datetime import datetime


class CLI(AbstractCLI):
    """CLI implements AbstractCLI from the cellbender package."""

    def __init__(self):
        self.name = 'remove-background'
        self.args = None

    def get_name(self) -> str:
        return self.name

    def validate_args(self, args):
        """Validate parsed arguments."""

        # Ensure that if there's a tilde for $HOME in the file path, it works.
        try:
            args.input_file = os.path.expanduser(args.input_file)
            args.output_file = os.path.expanduser(args.output_file)
        except TypeError:
            raise ValueError("Problem with provided input and output paths.")

        # Ensure write access to the save directory.
        file_dir, _ = os.path.split(args.output_file)  # Get the output directory
        if file_dir:
            assert os.access(file_dir, os.W_OK), \
                f"Cannot write to specified output directory {file_dir}. " \
                f"Ensure the directory exists and is write accessible."

        # If cell counts are specified, it should be positive.
        if args.expected_cell_count is not None:
            assert args.expected_cell_count > 0, \
                "expected_cells must be an integer greater than zero."

        # If additional barcodes are specified, they must be specified for
        # all files.
        if ((args.total_droplets is not None)
                and (args.expected_cell_count is not None)):
            assert args.total_droplets > args.expected_cell_count, \
                f"total_droplets must be an integer greater than the input " \
                f"expected_cell_count, which is {args.expected_cell_count}."

        assert (args.fraction_empties > 0) and (args.fraction_empties < 1), \
            "fraction_empties must be between 0 and 1, exclusive.  This is " \
            "the fraction of each minibatch that is composed of empty droplets."

        assert args.learning_rate < 0.1, "learning-rate must be < 0.1"
        assert args.learning_rate > 0, "learning-rate must be > 0"

        # Set training_fraction to consts.TRAINING_FRACTION (which is 1.).
        # args.training_fraction = consts.TRAINING_FRACTION
        assert args.training_fraction > 0, "training-fraction must be > 0"
        assert args.training_fraction <= 1., "training-fraction must be <= 1"

        # If cuda is requested, make sure it is available.
        if args.use_cuda:
            assert torch.cuda.is_available(), "Trying to use CUDA, " \
                                              "but CUDA is not available."
        else:
            # Warn the user in case the CUDA flag was forgotten by mistake.
            if torch.cuda.is_available():
                sys.stdout.write("Warning: CUDA is available, but will not be "
                                 "used.  Use the flag --cuda for "
                                 "significant speed-ups.\n\n")
                sys.stdout.flush()  # Write immediately

        # Ensure all network layer dimensions are positive.
        for n in args.z_hidden_dims:
            assert n > 0, "--z-layers must be all positive integers."

        # Ensure that z_hidden_dims are in encoder order.
        # (The same dimensions are used in reverse order for the decoder.)
        args.z_hidden_dims = sorted(args.z_hidden_dims, reverse=True)

        # Set use_jit to False.
        args.use_jit = False

        # Ensure false positive rate is between zero and one.
        for fpr in args.fpr:
            assert (fpr > 0.) and (fpr < 1.), \
                "False positive rate --fpr must be between 0 and 1."

        self.args = args

        return args

    def run(self, args):
        """Run the main tool functionality on parsed arguments."""

        # Run the tool.
        main(args)


def run_remove_background(args):
    """The full script for the command line tool to remove background RNA.

    Args:
        args: Inputs from the command line, already parsed using argparse.

    Note: Returns nothing, but writes output to a file(s) specified from
        command line.

    """

    # Load dataset, run inference, and write the output to a file.

    # Send logging messages to stdout as well as a log file.
    file_dir, file_base = os.path.split(args.output_file)
    file_name = os.path.splitext(os.path.basename(file_base))[0]
    log_file = os.path.join(file_dir, file_name + ".log")
    logging.basicConfig(level=logging.INFO,
                        format="cellbender:remove-background: %(message)s",
                        filename=log_file,
                        filemode="w")
    console = logging.StreamHandler()
    formatter = logging.Formatter("cellbender:remove-background: "
                                  "%(message)s")
    console.setFormatter(formatter)  # Use the same format for stdout.
    logging.getLogger('').addHandler(console)  # Log to stdout and a file.

    # Log the command as typed by user.
    logging.info("Command:\n" + ' '.join(['cellbender', 'remove-background']
                                         + sys.argv[2:]))

    # Log the start time.
    logging.info(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    logging.info("Running remove-background")

    # Load data from file and choose barcodes and genes to analyze.
    try:
        dataset_obj = \
            SingleCellRNACountsDataset(input_file=args.input_file,
                                       expected_cell_count=args.expected_cell_count,
                                       total_droplet_barcodes=args.total_droplets,
                                       fraction_empties=args.fraction_empties,
                                       model_name=args.model,
                                       gene_blacklist=args.blacklisted_genes,
                                       exclude_antibodies=args.exclude_antibodies,
                                       low_count_threshold=args.low_count_threshold,
                                       fpr=args.fpr)

    except OSError:
        logging.error(f"OSError: Unable to open file {args.input_file}.")
        sys.exit(1)

    # Instantiate latent variable model and run full inference procedure.
    inferred_model = run_inference(dataset_obj, args)

    # Write outputs to file.
    try:
        dataset_obj.save_to_output_file(args.output_file,
                                        inferred_model,
                                        save_plots=True)

        logging.info("Completed remove-background.")
        logging.info(datetime.now().strftime('%Y-%m-%d %H:%M:%S\n'))

    # The exception allows user to end inference prematurely with CTRL-C.
    except KeyboardInterrupt:

        # If partial output has been saved, delete it.
        full_file = args.output_file

        # Name of the filtered (cells only) file.
        file_dir, file_base = os.path.split(full_file)
        file_name = os.path.splitext(os.path.basename(file_base))[0]
        filtered_file = os.path.join(file_dir,
                                     file_name + "_filtered.h5")

        if os.path.exists(full_file):
            os.remove(full_file)

        if os.path.exists(filtered_file):
            os.remove(filtered_file)

        logging.info("Keyboard interrupt.  Terminated without saving.\n")


def main(args):
    """Take command-line input, parse arguments, and run tests or tool."""

    # Run the tool.
    run_remove_background(args)
