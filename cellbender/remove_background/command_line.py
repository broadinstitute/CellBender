"""Command-line tool functionality."""

import torch
from cellbender.remove_background.data.dataset import Dataset
import cellbender.remove_background.data.transform as transform
from cellbender.remove_background.train import run_inference
import cellbender.remove_background.tests.test
from cellbender.command_line import AbstractCLI

import argparse
import logging
import os
import sys
import unittest


class CLI(AbstractCLI):
    """CLI implements AbstractCLI from the cellbender package."""

    def __init__(self):
        self.name = 'remove_background'
        self.args = None

    def get_name(self) -> str:
        return self.name

    def add_subparser_args(self, subparsers: argparse) -> argparse:
        """Add tool-specific arguments for remove_background.

        Args:
            subparsers: Parser object before addition of arguments specific to
                remove_background.

        Returns:
            parser: Parser object with additional parameters.

        """

        subparser = subparsers.add_parser(self.name,
                                          description="Remove background RNA from "
                                                      "count matrix.",
                                          help="Remove background ambient RNA and "
                                               "barcode-swapped reads from a count "
                                               "matrix, producing a new count matrix "
                                               "and determining which barcodes "
                                               "contain real cells.")

        subparser.add_argument("--input", nargs="+", type=str,
                               dest='input_files', default=[],
                               help="Data files on which to run tool, as "
                                    "_raw_gene_barcode_matrices_h5.h5 files.")
        subparser.add_argument("--output", nargs="+", type=str,
                               dest='output_files', default=[],
                               help="Path to location of output data files, which "
                                    "will be .h5 files.")
        subparser.add_argument("--epochs", type=int, default=150,
                               dest="epochs",
                               help="Number of epochs to train.")
        subparser.add_argument("--z_dim", type=int, default=20,
                               dest="z_dim",
                               help="Dimension of latent variable z.")
        subparser.add_argument("--z_layers", nargs="+", type=int, default=[500],
                               dest="z_hidden_dims",
                               help="Dimension of hidden layers in the encoder for z.")
        subparser.add_argument("--d_layers", nargs="+", type=int, default=[5, 2, 2],
                               dest="d_hidden_dims",
                               help="Dimension of hidden layers in the encoder for d.")
        subparser.add_argument("--p_layers", nargs="+", type=int, default=[100, 10],
                               dest="p_hidden_dims",
                               help="Dimension of hidden layers in the encoder for p.")
        subparser.add_argument("--expected_cells", nargs="+", type=int, default=[None],
                               dest="expected_cell_count",
                               help="Number of cells expected in each dataset.")
        subparser.add_argument("--additional_barcodes", nargs="+", type=int,
                               default=[None], dest="additional_barcodes",
                               help="The number of additional droplets after the "
                                    "number of expected cells "
                                    "that have any chance of containing cells.")
        subparser.add_argument("--empty_drop_training_fraction",
                               type=float, default=0.5,
                               dest="fraction_empties",
                               help="Fraction of the training data that should be "
                                    "empty droplets.")
        subparser.add_argument("--blacklist_genes", nargs="+", type=int, default=[],
                               dest="blacklisted_genes",
                               help="Indices of genes to ignore entirely.")
        subparser.add_argument("--model", nargs="+", type=str, default=["full"],
                               choices=["simple", "ambient",
                                        "swapping", "full"],
                               dest="model",
                               help="Which model is being used for count data.  "
                                    "'simple' does not model background or ambient "
                                    "RNA.  'ambient' assumes background RNA is "
                                    "incorporated into droplets.  'swapping' assumes "
                                    "background RNA comes from barcode swapping. "
                                    "'full' uses a combined ambient and swapping "
                                    "model.")
        subparser.add_argument("--transform_counts", nargs=1, type=str,
                               default=["identity"],
                               choices=["identity", "log", "sqrt"],
                               dest="transform",
                               help="Transformation to apply to count data prior "
                                    "to running inference.")
        subparser.add_argument("--learning_rate", type=float, default=1e-3,
                               dest="learning_rate",
                               help="Learning rate for inference (probably don't "
                                    "exceed 1e-3).")
        # subparser.add_argument("--lambda", type=float, default=0.,
        #                        dest="lambda_reg",
        #                        help="Regularization parameter.  Default zero.  "
        #                             "L1 regularizer applied to decoder.")
        subparser.add_argument("--cuda",
                               dest="use_cuda", action="store_true",
                               help="Including the flag --cuda will run the inference "
                                    "on a GPU.")
        subparser.add_argument("--decaying_average_baseline",
                               dest="use_decaying_average_baseline",
                               action="store_true",
                               help="Including the flag "
                                    "--decaying_average_baseline will use a "
                                    "decaying average baseline during inference.")
        subparser.add_argument("--low_count_threshold", type=int, default=30,
                               dest="low_count_threshold",
                               help="Droplets with UMI counts below this are"
                                    "completely excluded from the analysis.  This "
                                    "can help adjust the prior for empty droplet "
                                    "counts in the rare case where empty counts "
                                    "are extremely high (over 200).")
        subparser.add_argument("--test",
                               dest="test", action="store_true",
                               help="Including the flag --test will run tests only, "
                                    "and disregard other input parameters.")

        return subparsers

    def validate_args(self, args):
        """Validate parsed arguments."""

        # Ensure write access to the save directory.
        for file in args.output_files:
            file_dir, _ = os.path.split(file)  # Get the output directory
            if file_dir:
                assert os.access(file_dir, os.W_OK), \
                    f"Cannot write to specified output directory {file_dir}"

        # If running tests, skip the rest of the validation.
        if args.test:
            self.args = args
            return args

        # If not testing, there must be at least one input and output file.
        assert len(args.input_files) > 0, "Specify data files using the " \
                                          "--input_files flag."

        # Number of input files must match number of output files.
        assert len(args.input_files) == len(args.output_files), \
            "Must specify the same number of input and output files."

        # If there is more than one model specified, the number must match files.
        if len(args.model) > 1:
            assert len(args.input_files) == len(args.model), \
                "If more than one model is specified, the number of models must" \
                "match the number of files, as they are paired."

        # If cell counts are specified, they must be specified for all files.
        if args.expected_cell_count[0] is not None:
            assert len(args.expected_cell_count) == len(args.input_files), \
                "Must either specify expected_cells for each input file, " \
                "or for none of them."

        # If additional barcodes are specified, they must be specified for all files.
        if args.additional_barcodes[0] is not None:
            assert len(args.additional_barcodes) == len(args.input_files), \
                "Must either specify additional_barcodes for each input file, " \
                "or for none of them."

        assert (args.fraction_empties > 0) and (args.fraction_empties < 1), \
            "fraction_empties must be between 0 and 1, exclusive.  This is the " \
            "fraction of each minibatch that is composed of empty droplets."

        assert args.learning_rate < 0.1, "learning_rate must be < 0.1"
        assert args.learning_rate > 0, "learning_rate must be > 0"

        # If cuda is requested, make sure it is available.
        if args.use_cuda:
            assert torch.cuda.is_available(), "Trying to use CUDA, " \
                                              "but CUDA is not available."
        else:
            # Warn the user in case the CUDA flag was forgotten by mistake.
            if torch.cuda.is_available():
                sys.stdout.write("Warning: CUDA is available, but will not be used.  "
                                 "Use the flag --cuda for significant speed-ups.\n\n")
                sys.stdout.flush()  # Write immediately

        # Ensure that z_hidden_dims are in encoder order.
        # (The same dimensions are used in reverse order for the decoder.)
        args.z_hidden_dims = sorted(args.z_hidden_dims, reverse=True)

        # Ensure that if there's a tilde for $HOME in the file path, it works.
        for i in range(len(args.input_files)):
            args.input_files[i] = os.path.expanduser(args.input_files[i])
            args.output_files[i] = os.path.expanduser(args.output_files[i])

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

    Note: Returns nothing, but writes output to a file(s) specified from command
    line.

    """

    # Set up the count data transformation.
    if args.transform[0] == "identity":
        trans = transform.IdentityTransform()
    elif args.transform[0] == "log":
        trans = transform.LogTransform(scale_factor=1.)
    elif args.transform[0] == "sqrt":
        trans = transform.SqrtTransform(scale_factor=1.)
    else:
        raise NotImplementedError(f"transform was set to {args.transform[0]}, "
                                  f"which is not implemented.")

    # If one model / cell-count specified for several files, broadcast it.
    if len(args.model) == 1:
        args.model = [args.model[0] for _ in range(len(args.input_files))]
    if len(args.expected_cell_count) == 1:
        args.expected_cell_count = \
            [args.expected_cell_count[0] for _ in range(len(args.input_files))]
    if len(args.additional_barcodes) == 1:
        args.additional_barcodes = \
            [args.additional_barcodes[0] for _ in range(len(args.input_files))]

    # Load each dataset, run inference, and write the output to a file.
    for i, file in enumerate(args.input_files):

        # Send logging messages to stdout as well as a log file.
        # TODO: this doesn't work for multiple file batching.
        file_dir, file_base = os.path.split(args.output_files[i])
        file_name = os.path.splitext(os.path.basename(file_base))[0]
        log_file = os.path.join(file_dir, file_name + ".log")
        logging.basicConfig(level=logging.INFO,
                            format="cellbender:remove_background: %(message)s",
                            filename=log_file,
                            filemode="w")
        console = logging.StreamHandler()
        formatter = logging.Formatter("cellbender:remove_background: %(message)s")
        console.setFormatter(formatter)  # Use the same format for stdout.
        logging.getLogger('').addHandler(console)  # Log to stdout as well as file.

        logging.info("Running remove_background")

        # Load data from file and choose barcodes and genes to analyze.
        try:
            dataset_obj = Dataset(transformation=trans,
                                  input_file=file,
                                  expected_cell_count=args.expected_cell_count[i],
                                  num_transition_barcodes=args.additional_barcodes[i],
                                  fraction_empties=args.fraction_empties,
                                  model_name=args.model[i],
                                  gene_blacklist=args.blacklisted_genes,
                                  low_count_threshold=args.low_count_threshold)
        except OSError:
            logging.error(f"OSError: Unable to open file {file}.")
            continue

        # Instantiate latent variable model and run full inference procedure.
        inferred_model = run_inference(dataset_obj, args)

        # Write outputs to file.
        try:
            dataset_obj.save_to_output_file(args.output_files[i], inferred_model,
                                            save_plots=True)

            logging.info("Completed remove_background.\n")

        # The exception allows user to end inference prematurely with CTRL-C.
        except KeyboardInterrupt:

            # If partial output has been saved, delete it.
            full_file = args.output_files[i]

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

        # Save trained model to file with same filename, but as .model.
        # file_dir, file_base = os.path.split(args.output_files[i])
        # file_name = os.path.splitext(os.path.basename(file_base))[0]
        # inferred_model.save_model_to_file(os.path.join(file_dir, file_name))
        del dataset_obj
        del inferred_model


def main(args):
    """Take command-line input, parse arguments, and run tests or tool."""

    # Figure out whether we are just running tests.
    testing = args.test

    if testing:

        # Run tests.
        cellbender.remove_background.tests.test.main()

    else:

        # Run the full tool.
        run_remove_background(args)
