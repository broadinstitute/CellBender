.. _remove background tutorial:

remove-background
=================

In this tutorial, we will run ``remove-background`` on a small dataset derived from the 10x Genomics
``pbmc4k`` scRNA-seq dataset (v2 Chemistry, CellRanger 2.1.0).

Trim PBMC dataset
-----------------

As a first step, we download the full dataset and generate a smaller `trimmed` copy by selecting 500 barcodes
with high UMI count (likely non-empty) and an additional 50'000 barcodes with small UMI count (likely empty). Note
that the trimming step is performed in order to allow us go through this tutorial in a matter of minutes on a
typical personal computer. Processing the full untrimmed dataset requires a CUDA-enabled GPU (e.g. NVIDIA Testla K80)
and takes about 30 minutes to finish.

We have created a python script to download and trim the dataset. Navigate to ``examples/remove_background/``
under your CellBender installation root directory and run the following command in the console:

.. code-block:: bash

   $ python generate_tiny_10x_pbmc.py

After successful completion of the script, you should have a new directory named ``tiny_raw_gene_bc_matrices``
containing ``GRCh38/matrix.mtx``, ``GRCh38/genes.tsv``, and ``GRCh38/barcodes.tsv``.

Run ``remove-background``
-------------------------

We proceed to run ``remove-background`` on the trimmed dataset using the following command:

.. code-block:: bash

   $ cellbender remove-background \
        --input ./tiny_raw_gene_bc_matrices/GRCh38 \
        --output ./tiny_10x_pbmc.h5 \
        --expected-cells 500 \
        --total-droplets-included 5000

Outputs
-------

The computation will finish within a minute or two (after ~ 150 epochs). The tool outputs the following files:

* ``tiny_10x_pbmc.h5``: An HDF5 file containing a detailed output of the inference procedure, including the
  normalized abundance of ambient transcripts, contamination fraction of each droplet, a low-dimensional
  embedding of the background-corrected gene expression, and the background-corrected counts matrix (in CSC sparse
  format). Please refer to the full documentation for a detailed description of these and other fields.

* ``tiny_10x_pbmc_filtered.h5``: Same as above, though, only including droplets with a posterior cell probability
  exceeding 0.5.

* ``tiny_10x_pbmc_cell_barcodes.csv``: The list of barcodes with a posterior cell probability exceeding 0.5.

* ``tiny_10x_pbmc.pdf``: A PDF summary of the results showing (1) the evolution of the loss function during training,
  (2) a ranked-ordered total UMI plot along with posterior cell probabilities, and (3) a two-dimensional PCA
  scatter plot of the latent embedding of the expressions in cell-containing droplets. Notice the rapid drop in
  the cell probability after UMI rank ~ 500.

Finally, try running the tool with ``--expected-cells 100`` and ``--expected-cells 1000``. You should find that
the output remains virtually the same.

Use output count matrix in downstream analyses
----------------------------------------------

The count matrix that ``remove-background`` generates can be easily loaded and used for downstream analyses in
`scanpy <https://scanpy.readthedocs.io/>`_ and `Seurat <https://satijalab.org/seurat/>`_.

To load the filtered count matrix (containing only cells) into scanpy:

.. code-block:: python

   # import scanpy
   import scanpy as sc

   # load the data
   adata = sc.read_10x_h5('tiny_10x_pbmc_filtered.h5', genome='background_removed')

To load the filtered count matrix (containing only cells) into Seurat:

.. code-block:: rd

   # load Seurat (version 3) library
   library(Seurat)

   # load data from the filtered h5 file
   data.file <- 'tiny_10x_pbmc_filtered.h5'
   data.data <- Read10X_h5(filename = data.file, use.names = TRUE)

   # create Seurat object
   obj <- CreateSeuratObject(counts = data.data)

Use latent gene expression in downstream analyses
-------------------------------------------------

To load the latent representation of gene expression ``z`` computed by ``remove-background`` in python:

.. code-block:: python

   import tables
   import numpy as np

   z = []
   with tables.open_file('tiny_10x_pbmc_filtered.h5') as f:
       print(f)  # display the structure of the h5 file
       z = f.root.background_removed.latent_gene_encoding.read()  # read latents

At this point, the variable ``z`` contains the latent encoding of gene expression, where rows are cells and
columns are dimensions of the latent variable.  This data can be saved in CSV format with the following command:

.. code-block:: python

   np.savetxt('tiny_10x_pbmc_latent_gene_expression.csv', z, delimiter=',')

This latent representation of gene expression can be loaded into a Seurat object ``obj`` by doing the following:

.. code-block:: rd

   # load the latent representation from cellbender
   latent <- read.csv('tiny_10x_pbmc_latent_gene_expression.csv', header = FALSE)
   latent <- t(data.matrix(latent))
   rownames(x = latent) <- paste0("CB", 1:20)
   colnames(x = latent) <- colnames(data.data)

   # store latent as a new dimensionality reduction called 'cellbender'
   obj[["cellbender"]] <- CreateDimReducObject(embeddings = t(latent),
                                               key = "CB_",
                                               assay = DefaultAssay(obj))

Or the variable ``z`` (from above) can be used directly in a scanpy ``anndata`` object.  The code snippet below
demonstrates loading the latent ``z`` and using it to do Louvain clustering:

.. code-block:: python

   # load the latent representation into a new slot called 'X_cellbender'
   adata.obsm['X_cellbender'] = z

   # perform louvain clustering using the cellbender latents and cosine distance
   sc.pp.neighbors(adata, use_rep='X_cellbender', metric='cosine')
   sc.pp.louvain(adata)
