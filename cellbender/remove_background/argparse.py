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
                                           "contain real cells.")

    subparser.add_argument("--input", nargs=None, type=str,
                           dest='input_file', default=None,
                           required=True,
                           help="Data file on which to run tool, as either "
                                "_raw_gene_barcode_matrices_h5.h5 file, or "
                                "as the path containing the raw "
                                "matrix.mtx, barcodes.tsv, and genes.tsv "
                                "files. Supported for outputs of "
                                "CellRanger v2 and v3.")
    subparser.add_argument("--output", nargs=None, type=str,
                           dest='output_file', default=None,
                           required=True,
                           help="Output file location (the path must "
                                "exist, and the file name must have .h5 "
                                "extension).")
    subparser.add_argument("--expected-cells", nargs=None, type=int,
                           default=None,
                           dest="expected_cell_count",
                           help="Number of cells expected in the dataset "
                                "(a rough estimate within a factor of 2 "
                                "is sufficient).")
    subparser.add_argument("--total-droplets-included",
                           nargs=None, type=int,
                           default=consts.TOTAL_DROPLET_DEFAULT,
                           dest="total_droplets",
                           help="The number of droplets from the "
                                "rank-ordered UMI plot that will be "
                                "analyzed. The largest 'total_droplets' "
                                "droplets will have their cell "
                                "probabilities inferred as an output.")
    subparser.add_argument("--model", nargs=None, type=str, default="full",
                           choices=["simple", "ambient",
                                    "swapping", "full"],
                           dest="model",
                           help="Which model is being used for count data. "
                                " 'simple' does not model either ambient "
                                "RNA or random barcode swapping (for "
                                "debugging purposes -- not recommended).  "
                                "'ambient' assumes background RNA is "
                                "incorporated into droplets.  'swapping' "
                                "assumes background RNA comes from random "
                                "barcode swapping.  'full' uses a "
                                "combined ambient and swapping model.  "
                                "Defaults to 'full'.")
    subparser.add_argument("--epochs", nargs=None, type=int, default=150,
                           dest="epochs",
                           help="Number of epochs to train.")
    subparser.add_argument("--cuda",
                           dest="use_cuda", action="store_true",
                           help="Including the flag --cuda will run the "
                                "inference on a GPU.")
    subparser.add_argument("--low-count-threshold", type=int,
                           default=consts.LOW_UMI_CUTOFF,
                           dest="low_count_threshold",
                           help="Droplets with UMI counts below this "
                                "number are completely excluded from the "
                                "analysis.  This can help identify the "
                                "correct prior for empty droplet counts "
                                "in the rare case where empty counts "
                                "are extremely high (over 200).")
    subparser.add_argument("--z-dim", type=int, default=20,
                           dest="z_dim",
                           help="Dimension of latent variable z.")
    subparser.add_argument("--z-layers", nargs="+", type=int, default=[500],
                           dest="z_hidden_dims",
                           help="Dimension of hidden layers in the encoder "
                                "for z.")
    subparser.add_argument("--d-layers", nargs="+", type=int,
                           default=[5, 2, 2],
                           dest="d_hidden_dims",
                           help="Dimension of hidden layers in the encoder "
                                "for d.")
    subparser.add_argument("--p-layers", nargs="+", type=int,
                           default=[100, 10],
                           dest="p_hidden_dims",
                           help="Dimension of hidden layers in the encoder "
                                "for p.")
    subparser.add_argument("--empty-drop-training-fraction",
                           type=float, nargs=None,
                           default=consts.FRACTION_EMPTIES,
                           dest="fraction_empties",
                           help="Training detail: the fraction of the "
                                "training data each epoch "
                                "that is drawn (randomly sampled) from "
                                "surely empty droplets.")
    subparser.add_argument("--blacklist-genes", nargs="+", type=int,
                           default=[],
                           dest="blacklisted_genes",
                           help="Integer indices of genes to ignore "
                                "entirely.  In the output count matrix, "
                                "the counts for these genes will be set "
                                "to zero.")
    subparser.add_argument("--learning-rate", nargs=None,
                           type=float, default=1e-3,
                           dest="learning_rate",
                           help="Training detail: learning rate for "
                                "inference (probably "
                                "do not exceed 1e-3).")

    return subparsers
