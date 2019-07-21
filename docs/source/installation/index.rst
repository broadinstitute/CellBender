.. _installation:

Installation and Usage
======================

Manual Installation
-------------------

The recommended installation is as follows. Create a conda environment and activate it:

.. code-block:: console

  $ conda create -n CellBender python=3.6
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

For `Terra <https://app.terra.bio>`_ users, a `workflow <https://portal.firecloud.org/#methods/broad-dsde-methods/cellbender/10/wdl>`_
is available as:

  ``broad-dsde-methods/cellbender``