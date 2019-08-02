.. _usage:

Usage
=====

remove-background
-----------------

Use case
~~~~~~~~

``remove-background`` is used to remove ambient / background RNA from a count matrix produced by
`10x Genomics' CellRanger pipeline
<https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/latest/what-is-cell-ranger>`_.
The output of ``cellranger count`` produces a raw .h5 file that is used as the input
for ``remove-background``.

``remove-background`` should be run on a dataset as a pre-processing step, before any downstream
analysis using Seurat, scanpy, your own custom analysis, etc.

The output of ``remove-background`` includes a new .h5 count matrix, with background RNA removed,
that can directly be used in downstream analysis in Seurat or scanpy as if it were the raw dataset.

A few caveats and hints:

* ``remove-background`` removes the background RNA that makes up the "ambient plateau": the same
  background RNA contained in empty droplets.  If your dataset has extremely few UMI counts in
  empty droplets, then there is not much background RNA present, and ``remove-background`` may
  not remove much.  See Exhibit A.
* If you have a dataset where you can identify an "empty droplet plateau" by eye, and these empty
  droplets have 50 or 100 or several hundred counts, then you can expect ``remove-background``
  to clean up your dataset significantly.  See Exhibit B.
* If you have a dataset with so much background RNA that you cannot identify the "empty droplet
  plateau" yourself by eye, then ``remove-background`` will also likely have a difficult time.
  Running the algorithm might be worth a try, but you should strongly consider re-running the experiment,
  as this points to a real QC failure.

.. image:: /_static/remove_background/UMI_curve_tropes.png
   :width: 800 px

Example
~~~~~~~

Pre-requisites:

* ``raw_feature_bc_matrix.h5``: A raw h5 count matrix file produced by `cellranger count
  <https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/3.1/what-is-cell-ranger>`_.
  (In CellRanger v2 this file was called ``raw_gene_bc_matrices_h5.h5``).

Run ``remove-background`` on the dataset using the following command
(leave out the flag ``--cuda`` if you are not using a GPU):

.. code-block:: console

   (CellBender) $ cellbender remove-background \
                    --input raw_feature_bc_matrix.h5 \
                    --output output.h5 \
                    --cuda \
                    --expected-cells 5000 \
                    --total-droplets-included 15000 \
                    --epochs 200

(The output filename "output.h5" can be replaced with a filename of choice.)

This command will produce five output files:

* ``output.h5``: Full count matrix as an h5 file, with background RNA removed.  This file
  contains all the original droplet barcodes.
* ``output_filtered.h5``: Filtered count matrix as an h5 file, with background RNA removed.
  The word "filtered" means that this file contains only the droplets which were
  determined to have a > 50% posterior probability of containing cells.
* ``output_cell_barcodes.csv``: CSV file containing all the droplet barcodes which were determined to have
  a > 50% posterior probability of containing cells.  Barcodes are written in plain text.
  This information is also contained in each of the above outputs, but is included as a separate
  output for convenient use in certain downstream applications.
* ``output.pdf``: PDF file that provides a standard graphical summary of the inference procedure.
* ``output.log``: Log file produced by the ``cellbender remove-background`` run.

Quality control checks
~~~~~~~~~~~~~~~~~~~~~~

* Check the log file for any warnings.
* Check lines 8 - 11 in the log file.  Ensure that the automatically-determined priors
  for cell counts and empty droplet counts match your expectation from the UMI curve.
  Ensure that the numbers of "probable cells", "additional barcodes", and "empty droplets"
  are all nonzero and look reasonable.
* Examine the PDF output.

    * Look at the upper plot to check whether
      it appears that the inference procedure has converged.  ``remove-background`` does not
      implement automatic early stopping, and it will not extend the number of epochs
      automatically.  If the value of the ELBO appears not to have converged to a reasonably
      stable value, then re-running with more epochs would be recommended.
    * Check the middle plot to see which droplets have been called as cells.  A converged
      inference procedure should result in the vast majority of cell probabilities
      being very close to either zero or one.  If the cell calls look problematic, check
      the :ref:`help documentation <remove background reference troubleshooting>`.
      Keep in mind that
      ``remove-background`` will output a high cell probability for any droplet that is
      unlikely to be drawn from the ambient background.  This can result in a large number
      of cells called.  The appropriate workflow would then be to filter cells downstream
      for things like mitochondrial read fraction.  This will remove some dying, low-expressing
      cells.
    * The lower plot shows a two-dimensional (PCA) projection of the inferred latent
      variable ``z`` that encodes gene expression.  Clusters in ``z``-space often
      correspond to different cell types.  If you see clustering in this plot, this is
      a good sign.  A lack of clustering could be due to only one cell type, or it could
      indicate QC problems with the dataset.  (For instance, if cells were all ruptured,
      all cells would appear to be the same "type".  This would coincide with
      difficulties in calling which droplets contain cells.)

Recommended best practices
~~~~~~~~~~~~~~~~~~~~~~~~~~

The default settings are good for getting started with a clean and simple dataset like
the publicly available `PBMC dataset from 10x Genomics
<https://support.10xgenomics.com/single-cell-gene-expression/datasets/2.1.0/pbmc8k>`_.
Only 150 epochs of training are necessary, and a low-capacity autoencoder is sufficient.

For datasets with more background RNA, and in order to do the least possible amount of
imputation, the following parameter settings are recommended:

* ``--epochs``: 300
* ``--expected-cells``: Base this on either the number of cells expected a priori from the
  experimental design, or if this is not known, base this number on the UMI curve as shown
  above.
* ``--total-droplets-included``: Choose a number that goes a few thousand barcodes into the "empty
  droplet plateau".  Include some droplets that you think are surely empty.  But be aware that
  the larger this number, the longer the algorithm takes to run (linear).  See the UMI curve
  below, where an appropriate choice would be 15,000.
  (This curve can be seen in the ``web_summary.html`` output from ``cellranger count``.)
* ``--cuda``: Include this flag.  The code is meant to be run on a GPU.
* ``--z-dim``: 200
* ``--z-layers``: 1000

.. image:: /_static/remove_background/UMI_curve_defs.png
   :width: 250 px
