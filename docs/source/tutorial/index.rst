.. _quick start tutorial:

Quick-start tutorial
====================

This section contains quick start tutorials for different CellBender modules.

.. _remove background tutorial:

remove-background
-----------------

In this tutorial, we will run ``remove-background`` on a small dataset derived from the 10x Genomics
``heart10k`` snRNA-seq `dataset
<https://www.10xgenomics.com/resources/datasets/10-k-heart-cells-from-an-e-18-mouse-v-3-chemistry-3-standard-3-0-0>`_
(v3 Chemistry, CellRanger 3.0.2).

As a first step, we download the full dataset and generate a smaller `trimmed` copy by selecting 500 barcodes
with high UMI count (likely non-empty) and an additional 50,000 barcodes with small UMI count (likely empty).
We also trim to keep only the top 100 most highly-expressed genes.  Note
that the trimming step is performed in order to allow us go through this tutorial in a minute on a
typical CPU. Processing the full untrimmed dataset requires a CUDA-enabled GPU (e.g. NVIDIA Testla T4)
and takes about 30 minutes to finish.

(Please note that trimming is NOT part of the recommended workflow, and is only for
the purposes of a quick demo run. When you run ``remove-background`` on real data,
do not do any preprocessing or feature selection or barcode selection first.)

We have created a python script to download and trim the dataset. Navigate to ``examples/remove_background/``
under your CellBender installation root directory and run the following command in the console:

.. code-block:: console

   $ python generate_tiny_10x_dataset.py

After successful completion of the script, you should have a new file named
``tiny_raw_feature_bc_matrix.h5ad``.

Run remove-background
~~~~~~~~~~~~~~~~~~~~~

We proceed to run ``remove-background`` on the trimmed dataset using the following command:

.. code-block:: console

   (cellbender) $ cellbender remove-background \
        --input tiny_raw_feature_bc_matrix.h5ad \
        --output tiny_output.h5 \
        --expected-cells 500 \
        --total-droplets-included 2000

Again, here we leave out the ``--cuda`` flag solely for the purposes of being able to run this
command on a CPU.  But a GPU is highly recommended for real datasets.

The computation will finish within a minute or two (after 150 epochs). The tool outputs the following files:

* ``tiny_output.h5``: An HDF5 file containing a detailed output of the inference procedure, including the
  normalized abundance of ambient transcripts, contamination fraction of each droplet, a low-dimensional
  embedding of the background-corrected gene expression, and the background-corrected counts matrix (in CSC sparse
  format). Please refer to the full documentation for a detailed description of these and other fields.

* ``tiny_output_filtered.h5``: Same as above, though, only including droplets with a posterior cell probability
  exceeding 0.5.

* ``tiny_output_cell_barcodes.csv``: The list of barcodes with a posterior cell probability exceeding 0.5.

* ``tiny_output.pdf``: A PDF summary of the results showing (1) the evolution of the loss function during training,
  (2) a ranked-ordered total UMI plot along with posterior cell probabilities, and (3) a two-dimensional PCA
  scatter plot of the latent embedding of the expressions in cell-containing droplets. Notice the rapid drop in
  the cell probability after UMI rank ~ 500.

* ``tiny_output_umi_counts.pdf``: A PDF showing the UMI counts per droplet as a histogram, annotated
  with what CellBender thinks is empty versus cells. This describes CellBender's prior. This is mainly
  a diagnostic if something seems to be going wrong with CellBender's automatic prior determination.

* ``tiny_output.log``: Log file.

* ``tiny_output_metrics.csv``: Output metrics, most of which have very descriptive names. This file is not
  used by most users, but the idea is that it can be incorporated into automated pipelines which could re-run
  CellBender automatically (with different parameters) if something goes wrong.

* ``tiny_output_report.html``: An HTML report which points out a few things about the run and
  highlights differences between the output and the input. Issues warnings if there are any
  aspects of the run that look anomalous, and makes suggestions.

Finally, try running the tool with ``--expected-cells 100`` and ``--expected-cells 1000``. You should find that
the output remains virtually the same.

.. _downstream-example:

Use output count matrix in downstream analyses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The count matrix that ``remove-background`` generates can be easily loaded and used for downstream analyses in
`scanpy <https://scanpy.readthedocs.io/>`_ and `Seurat <https://satijalab.org/seurat/>`_.

You can load the filtered count matrix (containing only cells) into scanpy:

.. code-block:: python

   # import scanpy
   import scanpy as sc

   # load the data
   adata = sc.read_10x_h5('tiny_output_filtered.h5')

This will yield ``adata`` like this

.. code-block:: console

   AnnData object with n_obs × n_vars = 531 × 100
       var: 'gene_ids', 'feature_types', 'genome'

The CellBender output counts are in ``adata.X``.

However, this does not include any of the CellBender metadata when loading
the file.  To include the metadata, but still load an ``AnnData`` object
that ``scanpy`` can operate on, try some of the functions from
``cellbender.remove_background.downstream`` (see :ref:`here <loading-outputs>`)

.. code-block:: python

   # import function
   from cellbender.remove_background.downstream import anndata_from_h5

   # load the data
   adata = anndata_from_h5('tiny_output.h5')

This yields an ``adata`` with all the cell barcodes which were analyzed by
CellBender (all the ``--total-droplets-included``), along with all the
metadata and latent variables inferred by CellBender:

.. code-block:: console

   AnnData object with n_obs × n_vars = 1000 × 100
       obs: 'background_fraction', 'cell_probability', 'cell_size', 'droplet_efficiency'
       var: 'ambient_expression', 'features_analyzed_inds', 'feature_type', 'genome', 'gene_id'
       uns: 'cell_size_lognormal_std', 'empty_droplet_size_lognormal_loc', 'empty_droplet_size_lognormal_scale', 'posterior_regularization_lambda', 'swapping_fraction_dist_params', 'target_false_positive_rate', 'fraction_data_used_for_testing', 'test_elbo', 'test_epoch', 'train_elbo', 'train_epoch'
       obsm: 'gene_expression_encoding'

(If you want to load both the
raw data and the CellBender data into one AnnData object, which is very useful,
try the ``load_anndata_from_input_and_output()`` function in
``cellbender.remove_background.downstream``, see :ref:`here <loading-outputs>`)

You can access the latent gene expression embedding learned by CellBender in
``adata.obsm['gene_expression_encoding']``, the inferred ambient RNA profile
is in ``adata.var['ambient_expression']``, and the inferred cell probabilties are
in ``adata.obs['cell_probability']``.

You can limit this ``adata`` to CellBender cell calls very easily:

.. code-block:: python

   adata[adata.obs['cell_probability'] > 0.5]

.. code-block:: console

   View of AnnData object with n_obs × n_vars = 531 × 100
       obs: 'background_fraction', 'cell_probability', 'cell_size', 'droplet_efficiency'
       var: 'ambient_expression', 'features_analyzed_inds', 'feature_type', 'genome', 'gene_id'
       uns: 'cell_size_lognormal_std', 'empty_droplet_size_lognormal_loc', 'empty_droplet_size_lognormal_scale', 'posterior_regularization_lambda', 'swapping_fraction_dist_params', 'target_false_positive_rate', 'fraction_data_used_for_testing', 'test_elbo', 'test_epoch', 'train_elbo', 'train_epoch'
       obsm: 'gene_expression_encoding'

How to use the latent gene expression downstream
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After loading data using the ``anndata_from_h5()`` function as shown above,
we can compute nearest neighbors
in scanpy, using the CellBender latent representation of cells, and make a UMAP and do clustering:

.. code-block:: python

   # compute a UMAP and do clustering using the cellbender latent gene expression embedding
   sc.pp.neighbors(adata, use_rep='gene_expression_encoding', metric='euclidean', method='umap')
   sc.pp.umap(adata)
   sc.pp.leiden(adata)

.. _open-in-seurat:

Seurat
~~~~~~

Seurat 4.0.2 uses a dataloader ``Read10X_h5()`` which is not currently compatible with
the CellBender output file format.  Hopefully Seurat will update its dataloader to
ignore extra information in the future, but in the interim, we can use a `super
handy utility from PyTables
<https://www.pytables.org/usersguide/utilities.html#ptrepack>`_ to strip the
extra CellBender information out of the output file so that Seurat can load it.

From a python environment in which PyTables is installed, do the following at
the command line:

.. code-block:: console

   $ ptrepack --complevel 5 tiny_output_filtered.h5:/matrix tiny_output_filtered_seurat.h5:/matrix

(The flag ``--complevel 5`` ensures that the file size does not increase.)

The file ``tiny_output_filtered_seurat.h5`` is now formatted *exactly* like
a CellRanger v3 h5 file, so Seurat can load it:

.. code-block:: Rd

   # load data from the filtered h5 file
   data.file <- 'tiny_output_filtered_seurat.h5'
   data.data <- Read10X_h5(filename = data.file, use.names = TRUE)

   # create Seurat object
   obj <- CreateSeuratObject(counts = data.data)
   obj

.. code-block:: console

   An object of class Seurat
   100 features across 531 samples within 1 assay
   Active assay: RNA (100 features, 0 variable features)

Of course, this will not load any metadata from CellBender, so if that is desired,
it would have to be accessed and added to the object another way.

Another option for loading data into Seurat would be third party packages like
`scCustomize from Samuel Marsh
<https://github.com/broadinstitute/CellBender/issues/145#issuecomment-1217360305>`_.
