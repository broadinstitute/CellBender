.. _installation:

Installation
============

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

Older versions are available at the same location, for example as

``us.gcr.io/broad-dsde-methods/cellbender:0.2.0``


Terra Workflow
--------------

For `Terra <https://app.terra.bio>`_ users (or any other users of WDL workflows),
the following WDL workflow is publicly available:

* `cellbender/remove-background <https://portal.firecloud.org/#methods/cellbender/remove-background/>`_

Some documentation for the WDL is available at the above link, and some is visible
`on github <https://github.com/broadinstitute/CellBender/tree/master/wdl>`_.
