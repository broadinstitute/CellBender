"""Command-line tool functionality for remove-background."""

import torch
import cellbender

from cellbender.base_cli import AbstractCLI, get_version
from cellbender.remove_background.checkpoint import create_workflow_hashcode
from cellbender.remove_background.run import run_remove_background
from cellbender.remove_background.posterior import Posterior

import logging
import os
import sys
import argparse


class CLI(AbstractCLI):
    """CLI implements AbstractCLI from the cellbender package."""

    def __init__(self):
        self.name = 'remove-background'
        self.args = None

    def get_name(self) -> str:
        return self.name

    @staticmethod
    def validate_args(args) -> argparse.Namespace:
        """Validate parsed arguments."""

        # Ensure that if there's a tilde for $HOME in the file path, it works.
        try:
            args.input_file = os.path.expanduser(args.input_file)
            args.output_file = os.path.expanduser(args.output_file)
            if args.truth_file is not None:
                args.truth_file = os.path.expanduser(args.truth_file)
        except TypeError:
            raise ValueError("Problem with provided input and output paths.")

        # Ensure that if truth data is specified, it is accessible
        if args.truth_file is not None:
            assert os.access(args.truth_file, os.R_OK), \
                f"Cannot read specified simulated truth file {args.truth_file}. " \
                f"Ensure the file exists and is read accessible."

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

        # Make sure n_threads makes sense.
        if args.n_threads is not None:
            assert args.n_threads > 0, "--cpu-threads must be an integer >= 1"

        # Ensure all network layer dimensions are positive.
        for n in args.z_hidden_dims:
            assert n > 0, "--z-layers must be all positive integers."

        # Ensure that z_hidden_dims are in encoder order.
        # (The same dimensions are used in reverse order for the decoder.)
        args.z_hidden_dims = sorted(args.z_hidden_dims, reverse=True)

        # Set use_jit to False.
        args.use_jit = False

        # Ensure false positive rate is between zero and one.
        fpr_list_correct_dtypes = []  # for a mix of floats and strings later on
        for fpr in args.fpr:
            try:
                fpr = float(fpr)
                assert (fpr >= 0.) and (fpr < 1.), \
                    "False positive rate --fpr must be in [0, 1)"
            except ValueError:
                # the input is not a float
                assert fpr == 'cohort', \
                    "The only allowed non-float value for FPR is the word 'cohort'."
            fpr_list_correct_dtypes.append(fpr)
        args.fpr = fpr_list_correct_dtypes

        # Ensure that "exclude_features" specifies allowed features.
        # As of CellRanger 7.2, the possible features are:
        #     Gene Expression
        #     Antibody Capture
        #     CRISPR Guide Capture
        #     Custom
        #     Peaks
        #     Multiplexing Capture
        #     VDJ
        #     VDJ-T
        #     VDJ-T-GD
        #     VDJ-B
        #     Antigen Capture
        allowed_features = ['Gene Expression', 'Antibody Capture',
                            'CRISPR Guide Capture', 'Custom', 'Peaks',
                            'Multiplexing Capture', 'VDJ', 'VDJ-T',
                            'VDJ-T-GD', 'VDJ-B', 'Antigen Capture']
        for feature in args.exclude_features:
            if feature not in allowed_features:
                sys.stdout.write(f"Specified '{feature}' using --exclude-feature-types, "
                                 f"but this is not a valid CellRanger feature "
                                 f"designation: {allowed_features}. Ensure that "
                                 f"this feature appears in your dataset, and "
                                 f"ensure that this log file makes note of the "
                                 f"exclusion of the appropriate features below.\n\n")
                sys.stdout.flush()  # Write immediately
        if 'Gene Expression' in args.exclude_features:
            sys.stdout.write("WARNING: Excluding 'Gene Expression' features from the analysis "
                             "is not recommended, since other features alone are typically "
                             "too sparse to form a good prior on cell type, and CellBender "
                             "relies on being able to construct this sort of prior\n\n")
            sys.stdout.flush()  # Write immediately

        # Automatic training failures and restarts.
        assert args.num_training_tries > 0, "--num-training-tries must be > 0 (default 1)"
        if args.epoch_elbo_fail_fraction is not None:
            assert (args.epoch_elbo_fail_fraction > 0.), \
                "--epoch-elbo-fail-fraction must be in > 0"
        if args.final_elbo_fail_fraction is not None:
            assert (args.final_elbo_fail_fraction > 0.), \
                "--final-elbo-fail-fraction must be in > 0"

        # Ensure timing of checkpoints is within bounds.
        assert args.checkpoint_min > 0, "--checkpoint-min must be > 0"
        if args.checkpoint_min > 15:
            sys.stdout.write(f"Warning: Timing between checkpoints is specified as "
                             f"{args.checkpoint_min} minutes.  Consider reducing "
                             f"this number if you are concerned about the "
                             f"possibility of lost work upon preemption.\n\n")
            sys.stdout.flush()  # Write immediately

        # Posterior regularization checking.
        if args.cdf_threshold_q is not None:
            assert (args.cdf_threshold_q >= 0.) and (args.cdf_threshold_q <= 1.), \
                f"Argument --q must be in range [0, 1] since it is a CDF threshold."
        if args.posterior_regularization == 'PRq':
            # We need q for the CDF threshold estimator.
            assert args.prq_alpha is not None, \
                'Input argument --alpha must be specified when using ' \
                '--posterior-regularization PRq'

        # Estimator checking.
        if args.estimator == 'cdf':
            # We need q for the CDF threshold estimator.
            assert args.cdf_threshold_q is not None, \
                'Input argument --q must be specified when using --estimator cdf'

        return args

    @staticmethod
    def run(args) -> Posterior:
        """Run the main tool functionality on parsed arguments."""

        # Run the tool.
        return main(args)


def setup_and_logging(args):
    """Take command-line input, parse arguments, and run tests or tool."""

    # Send logging messages to stdout as well as a log file.
    file_dir, file_base = os.path.split(args.output_file)
    file_name = os.path.splitext(os.path.basename(file_base))[0]
    log_file = os.path.join(file_dir, file_name + ".log")
    logger = logging.getLogger('cellbender')  # name of the logger
    logger.setLevel(logging.INFO if not args.debug else logging.DEBUG)
    formatter = logging.Formatter('cellbender:remove-background: %(message)s')
    file_handler = logging.FileHandler(filename=log_file, mode='w', encoding='UTF-8')
    console_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)  # set the file format
    console_handler.setFormatter(formatter)  # use the same format for stdout
    logger.addHandler(file_handler)  # log to file
    logger.addHandler(console_handler)  # log to stdout

    # Log the command as typed by user.
    logger.info("Command:\n"
                + ' '.join(['cellbender', 'remove-background'] + sys.argv[2:]))
    logger.info("CellBender " + get_version())

    return args, file_handler


def main(args) -> Posterior:
    """Take command-line input, parse arguments, and run tests or tool."""

    args, file_handler = setup_and_logging(args)

    # Run the tool.
    posterior = run_remove_background(args)
    file_handler.close()

    return posterior
