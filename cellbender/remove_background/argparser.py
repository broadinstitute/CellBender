import argparse
from cellbender.remove_background import consts


def add_subparser_args(subparsers: argparse) -> argparse:
    """Add tool-specific arguments for remove-background.

    Args:
        subparsers: Parser object before addition of arguments specific to
            remove-background.

    Returns:
        parser: Parser object with additional parameters.

    """

    subparser = subparsers.add_parser("remove-background",
                                      description="Remove background RNA "
                                                  "from count matrix.",
                                      help="Remove background ambient RNA "
                                           "and barcode-swapped reads from "
                                           "a count matrix, producing a "
                                           "new count matrix and "
                                           "determining which barcodes "
                                           "contain real cells.",
                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    subparser.add_argument("--input", nargs=None, type=str,
                           dest='input_file',
                           required=True,
                           help="Data file on which to run tool. Data must be "
                                "un-filtered: it should include empty droplets. "
                                "The following input formats are supported: "
                                "CellRanger v2 and v3 (.h5 or the directory that "
                                "contains the .mtx file), Dropseq "
                                "DGE (.txt or .txt.gz), BD Rhapsody (.csv or "
                                ".csv.gz), AnnData (.h5ad), and Loom (.loom).")
    subparser.add_argument("--output", nargs=None, type=str,
                           dest='output_file',
                           required=True,
                           help="Output file location (the path must "
                                "exist, and the file name must have .h5 "
                                "extension).")
    subparser.add_argument("--cuda",
                           dest="use_cuda", action="store_true",
                           help="Including the flag --cuda will run the "
                                "inference on a GPU.")
    subparser.add_argument("--checkpoint", nargs=None, type=str,
                           dest='input_checkpoint_tarball',
                           required=False, default=consts.CHECKPOINT_FILE_NAME,
                           help="Checkpoint tarball produced by v0.3.0+ "
                                "of CellBender remove-background.  If present, "
                                "and the workflow hashes match, training will "
                                "restart from this checkpoint.")
    subparser.add_argument("--force-use-checkpoint",
                           dest='force_use_checkpoint', action="store_true",
                           help="Normally, checkpoints can only be used if the CellBender "
                           "code and certain input args match exactly. This flag allows you "
                           "to bypass this requirement. An example use would be to create a new output "
                           "using a checkpoint from a run of v0.3.1, a redacted version with "
                           "faulty output count matrices. If you use this flag, "
                           "ensure that the input file and the checkpoint match, because "
                           "CellBender will not check.")
    subparser.add_argument("--expected-cells", nargs=None, type=int,
                           default=None,
                           dest="expected_cell_count",
                           help="Number of cells expected in the dataset "
                                "(a rough estimate within a factor of 2 "
                                "is sufficient).")
    subparser.add_argument("--total-droplets-included",
                           nargs=None, type=int, default=None,
                           dest="total_droplets",
                           help="The number of droplets from the "
                                "rank-ordered UMI plot that will have their "
                                "cell probabilities inferred as an output. "
                                "Include the droplets which might contain "
                                "cells. Droplets beyond TOTAL_DROPLETS_INCLUDED "
                                "should be 'surely empty' droplets.")
    subparser.add_argument("--force-cell-umi-prior",
                           nargs=None, type=float, default=None,
                           dest="force_cell_umi_prior",
                           help="Ignore CellBender's heuristic prior estimation, "
                                "and use this prior for UMI counts in cells.")
    subparser.add_argument("--force-empty-umi-prior",
                           nargs=None, type=float, default=None,
                           dest="force_empty_umi_prior",
                           help="Ignore CellBender's heuristic prior estimation, "
                                "and use this prior for UMI counts in empty droplets.")
    subparser.add_argument("--model", nargs=None, type=str,
                           default="full",
                           choices=["naive", "simple", "ambient", "swapping", "full"],
                           dest="model",
                           help="Which model is being used for count data. "
                                "'naive' subtracts the estimated ambient profile."
                                " 'simple' does not model either ambient "
                                "RNA or random barcode swapping (for "
                                "debugging purposes -- not recommended).  "
                                "'ambient' assumes background RNA is "
                                "incorporated into droplets.  'swapping' "
                                "assumes background RNA comes from random "
                                "barcode swapping (via PCR chimeras).  'full' "
                                "uses a combined ambient and swapping model.")
    subparser.add_argument("--epochs", nargs=None, type=int, default=150,
                           dest="epochs",
                           help="Number of epochs to train.")
    subparser.add_argument("--low-count-threshold", type=int,
                           default=consts.LOW_UMI_CUTOFF,
                           dest="low_count_threshold",
                           help="Droplets with UMI counts below this "
                                "number are completely excluded from the "
                                "analysis.  This can help identify the "
                                "correct prior for empty droplet counts "
                                "in the rare case where empty counts "
                                "are extremely high (over 200).")
    subparser.add_argument("--z-dim", type=int, default=64,
                           dest="z_dim",
                           help="Dimension of latent variable z.")
    subparser.add_argument("--z-layers", nargs="+", type=int, default=[512],
                           dest="z_hidden_dims",
                           help="Dimension of hidden layers in the encoder for z.")
    subparser.add_argument("--training-fraction",
                           type=float, nargs=None,
                           default=consts.TRAINING_FRACTION,
                           dest="training_fraction",
                           help="Training detail: the fraction of the "
                                "data used for training.  The rest is never "
                                "seen by the inference algorithm.  Speeds up "
                                "learning.")
    subparser.add_argument("--empty-drop-training-fraction",
                           type=float, nargs=None,
                           default=consts.FRACTION_EMPTIES,
                           dest="fraction_empties",
                           help="Training detail: the fraction of the "
                                "training data each epoch "
                                "that is drawn (randomly sampled) from "
                                "surely empty droplets.")
    subparser.add_argument("--ignore-features", nargs="+", type=int,
                           default=[],
                           dest="blacklisted_genes",
                           help="Integer indices of features to ignore "
                                "entirely.  In the output count matrix, the "
                                "counts for these features will be unchanged.")
    subparser.add_argument("--fpr", nargs="+",
                           default=[0.01],
                           dest="fpr",
                           help="Target 'delta' false positive rate in [0, 1). "
                                "Use 0 for a cohort of samples which will be "
                                "jointly analyzed for differential expression. "
                                "A false positive is a true signal count that is "
                                "erroneously removed.  More background removal "
                                "is accompanied by more signal removal "
                                "at high values of FPR.  You can specify "
                                "multiple values, which will create multiple "
                                "output files.")
    subparser.add_argument("--exclude-feature-types",
                           type=str, nargs="+", default=[],
                           dest="exclude_features",
                           help="Feature types to ignore during the analysis.  "
                                "These features will be left unchanged in the "
                                "output file.")
    subparser.add_argument("--projected-ambient-count-threshold",
                           type=float, nargs=None,
                           default=consts.AMBIENT_COUNTS_IN_CELLS_LOW_LIMIT,
                           dest="ambient_counts_in_cells_low_limit",
                           help="Controls how many features are included in the "
                                "analysis, which can lead to a large speedup. "
                                "If a feature is expected to have less than "
                                "PROJECTED_AMBIENT_COUNT_THRESHOLD counts total "
                                "in all cells (summed), then that gene is "
                                "excluded, and it will be unchanged in the "
                                "output count matrix.  For example, "
                                "PROJECTED_AMBIENT_COUNT_THRESHOLD = 0 will "
                                "include all features which have even a single "
                                "count in any empty droplet.")
    subparser.add_argument("--learning-rate",
                           type=float, default=1e-4,
                           dest="learning_rate",
                           help="Training detail: lower learning rate for "
                                "inference. A OneCycle learning rate schedule "
                                "is used, where the upper learning rate is ten "
                                "times this value. (For this value, probably "
                                "do not exceed 1e-3).")
    subparser.add_argument("--checkpoint-mins",
                           type=float, default=7.,
                           dest="checkpoint_min",
                           help="Checkpoint file will be saved periodically, "
                                "with this many minutes between each checkpoint.")
    subparser.add_argument("--final-elbo-fail-fraction", type=float,
                           dest="final_elbo_fail_fraction",
                           help="Training is considered to have failed if "
                                "(best_test_ELBO - final_test_ELBO)/(best_test_ELBO "
                                "- initial_test_ELBO) > FINAL_ELBO_FAIL_FRACTION.  Training will "
                                "automatically re-run if --num-training-tries > "
                                "1.  By default, will not fail training based "
                                "on final_training_ELBO.")
    subparser.add_argument("--epoch-elbo-fail-fraction", type=float,
                           dest="epoch_elbo_fail_fraction",
                           help="Training is considered to have failed if "
                                "(previous_epoch_test_ELBO - current_epoch_test_ELBO)"
                                "/(previous_epoch_test_ELBO - initial_train_ELBO) "
                                "> EPOCH_ELBO_FAIL_FRACTION.  Training will "
                                "automatically re-run if --num-training-tries > "
                                "1.  By default, will not fail training based "
                                "on epoch_training_ELBO.")
    subparser.add_argument("--num-training-tries", type=int, default=1,
                           dest="num_training_tries",
                           help="Number of times to attempt to train the model.  "
                                "At each subsequent attempt, the learning rate is "
                                "multiplied by LEARNING_RATE_RETRY_MULT.")
    subparser.add_argument("--learning-rate-retry-mult", type=float, default=0.2,
                           dest="learning_rate_retry_mult",
                           help="Learning rate is multiplied by this amount each "
                                "time a new training attempt is made.  (This "
                                "parameter is only used if training fails based "
                                "on EPOCH_ELBO_FAIL_FRACTION or "
                                "FINAL_ELBO_FAIL_FRACTION and NUM_TRAINING_TRIES"
                                " is > 1.)")
    subparser.add_argument("--posterior-batch-size", type=int,
                           default=consts.PROB_POSTERIOR_BATCH_SIZE,
                           dest="posterior_batch_size",
                           help="Training detail: size of batches when creating "
                                "the posterior.  Reduce this to avoid running "
                                "out of GPU memory creating the posterior "
                                "(will be slower).")
    subparser.add_argument("--posterior-regularization", type=str,
                           default=None,
                           choices=["PRq", "PRmu", "PRmu_gene"],
                           dest="posterior_regularization",
                           help="Posterior regularization method. (For experts: "
                                "not required for normal usage, see "
                                "documentation). PRq is approximate quantile-"
                                "targeting. PRmu is approximate mean-targeting "
                                "aggregated over genes (behavior of v0.2.0). "
                                "PRmu_gene is approximate mean-targeting per "
                                "gene.")
    subparser.add_argument("--alpha",
                           type=float, default=None,
                           dest="prq_alpha",
                           help="Tunable parameter alpha for the PRq posterior "
                                "regularization method (not normally used: see "
                                "documentation).")
    subparser.add_argument("--q",
                           type=float, default=None,
                           dest="cdf_threshold_q",
                           help="Tunable parameter q for the CDF threshold "
                                "estimation method (not normally used: see "
                                "documentation).")
    subparser.add_argument("--estimator", type=str,
                           default="mckp",
                           choices=["map", "mean", "cdf", "sample", "mckp"],
                           dest="estimator",
                           help="Output denoised count estimation method. (For "
                                "experts: not required for normal usage, see "
                                "documentation).")
    subparser.add_argument("--estimator-multiple-cpu",
                           dest="use_multiprocessing_estimation", action="store_true",
                           default=False,
                           help="Including the flag --estimator-multiple-cpu will "
                                "use more than one CPU to compute the MCKP "
                                "output count estimator in parallel (does "
                                "nothing for other estimators).")
    subparser.add_argument("--constant-learning-rate",
                           dest="constant_learning_rate", action="store_true",
                           default=False,
                           help="Including the flag --constant-learning-rate will "
                                "use the ClippedAdam optimizer instead of the "
                                "OneCycleLR learning rate schedule, which is "
                                "the default.  Learning is faster with the "
                                "OneCycleLR schedule.  However, training can "
                                "easily be continued from a checkpoint for more "
                                "epochs than the initial command specified when "
                                "using ClippedAdam.  On the other hand, if using "
                                "the OneCycleLR schedule with 150 epochs "
                                "specified, it is not possible to pick up from "
                                "that final checkpoint and continue training "
                                "until 250 epochs.")
    subparser.add_argument("--cpu-threads",
                           type=int, default=None,
                           dest="n_threads",
                           help="Number of threads to use when pytorch is run "
                                "on CPU. Defaults to the number of logical cores.")
    subparser.add_argument("--debug",
                           dest="debug", action="store_true",
                           help="Including the flag --debug will log "
                                "extra messages useful for debugging.")
    subparser.add_argument("--truth",
                           type=str, default=None,
                           dest="truth_file",
                           help="This is only used by developers for report "
                                "generation.  Truth h5 file (for "
                                "simulated data only).")

    return subparsers
