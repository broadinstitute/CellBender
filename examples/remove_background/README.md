### Example usage of `remove-background` module

In this tutorial, we will run `cellbender remove-background` on a small dataset derived from the 10x Genomics
`pbmc4k` scRNA-seq dataset (v2 Chemistry, CellRanger 2.1.0).

As a first step, we download the full dataset and generate a smaller _trimmed_ copy by selecting 500 barcodes
with high UMI count (likely non-empty) and an additional 50'000 barcodes with small UMI count (likely empty). Note
that the trimming step is performed in order to allow us go through this tutorial in a matter of minutes on a
typical personal computer. Processing the full untrimmed dataset requires a CUDA-enabled GPU (e.g. NVIDIA Testla K80)
and takes about 30 minutes to finish.

We have created a convenient python script to download and trim the dataset:
```
python generate_tiny_10x_pbmc.py
```

After successful completion of the script, you should have a new directory named `tiny_raw_gene_bc_matrices`
containing `GRCh38/matrix.mtx`, `GRCh38/genes.tsv`, and `GRCh38/barcodes.tsv`.

We proceed to run `remove-background` on the trimmed dataset using the following command:
```
cellbender remove-background \
    --input ./tiny_raw_gene_bc_matrices/GRCh38 \
    --output ./tiny_10x_pbmc.h5 \
    --expected-cells 500 \
    --total-droplets-included 5000
```

The computation will finish within a minute or two (after ~ 150 epochs). The tool outputs the following files:

* `tiny_10x_pbmc.h5`: An HDF5 file containing a detailed output of the inference procedure, including the
normalized abundance of ambient transcripts, contamination fraction of each droplet, a low-dimensional
embedding of the background-corrected gene expression, and the background-corrected counts matrix (in CSC sparse
format). Please refer to the full documentation for a detailed description of these and other fields.

* `tiny_10x_pbmc_filtered.h5`: Same as above, though, only including droplets with a posterior cell probability
exceeding 0.5.

* `tiny_10x_pbmc_cell_barcodes.csv`: The list of barcodes with a posterior cell probability exceeding 0.5.

* `tiny_10x_pbmc.pdf`: A PDF summary of the results showing (1) the evolution of the loss function during training,
(2) a ranked-ordered total UMI plot along with posterior cell probabilities, and (3) a two-dimensional PCA
scatter plot of the latent embedding of the expressions in cell-containing droplets. Notice the rapid drop in
the cell probability after UMI rank ~ 500.

Finally, try running the tool with `--expected-cells 100` and `--expected-cells 1000`. You should find that
the output remains virtually the same.
