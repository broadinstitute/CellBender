CellBender
==========

Python package for analysis of single-cell RNA sequencing data.

remove_background
-----------------

Transcript counts are the sum of real counts and some background
counts from ambient RNA.
The purpose of the remove_background tool is to remove the background RNA counts
from real cells, and to determine which barcodes
contain cells and which correspond to "empty" droplets with
only ambient RNA.  The inference procedure also embeds gene
expression in a lower-dimensional latent space, useful for
clustering and visualization.

Input:

Input a raw count matrix from single cell RNA sequencing,
including cell barcodes and non-cell barcodes alike.

Outputs:

The output of the analysis is a new count matrix, which is the
counts after background subtraction.
A second output is the probability that each barcode contains a real cell.
A third output is a low-dimensional latent representation of the gene
expression of each cell.

The background-subtracted count matrix as well as the cell probabilities
can be used in downstream analyses.  The low-dimensional latent
representation can be used for visualization and clustering, as
well as analyses downstream of clustering.

Methodology:

This is a principled approach to data analysis based on inference in
the context of a generative probabilistic model of single-cell RNA
sequencing data.  An autoencoder setup is used to implement
variational Bayesian inference within the framework of our model.

Under the hood, this analysis runs on pyro for the variational
inference architecture and pytorch for gradient-based optimization.

