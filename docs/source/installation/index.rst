.. _installation:

Installation and Usage
======================

Manual Installation
-------------------

The recommended installation is as follows. Create a conda environment and activate it:

.. code-block:: console

  $ conda create -n CellBender python=3.7
  $ source activate CellBender

Install the `pytables <https://www.pytables.org>`_ module:

.. code-block:: console

  (CellBender) $ conda install -c anaconda pytables

Install `pytorch <https://pytorch.org>`_ (shown below for CPU; if you have a CUDA-ready GPU, please skip
this part and follow `these instructions <https://pytorch.org/get-started/locally/>`_ instead):

.. code-block:: console

   (CellBender) $ conda install pytorch torchvision -c pytorch

Clone this repository and install CellBender:

.. code-block:: console

   (CellBender) $ git clone https://github.com/broadinstitute/CellBender.git
   (CellBender) $ pip install -e CellBender


Docker Image
------------

A GPU-enabled docker image is available from the Google Container Registry (GCR) as:

``us.gcr.io/broad-dsde-methods/cellbender:latest``


Terra Workflow
--------------

For `Terra <https://app.terra.bio>`_ users, the following workflow is publicly available:

* `cellbender/remove-background <https://portal.firecloud.org/#methods/cellbender/remove-background/>`_

Usage
-----

remove-background
~~~~~~~~~~~~~~~~~

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
                    --total-droplets-included 15000

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
