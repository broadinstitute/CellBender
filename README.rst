CellBender
==========

.. image:: https://readthedocs.org/projects/cellbender/badge/?version=latest
   :target: https://cellbender.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://github.com/broadinstitute/CellBender/blob/master/docs/source/_static/design/logo_250_185.png
   :alt: CellBender Logo

CellBender is a software package for eliminating technical artifacts from
high-throughput single-cell RNA sequencing (scRNA-seq) data.

The current release contains the following modules. More modules will be added in the future:

* ``remove-background``:

  This module removes counts due to ambient RNA molecules and random barcode swapping from (raw)
  UMI-based scRNA-seq count matrices. At the moment, only the count matrices produced by the
  CellRanger ``count`` pipeline is supported. Support for additional tools and protocols will be
  added in the future. A quick start tutorial can be found
  `here <https://cellbender.readthedocs.io/en/latest/getting_started/remove_background/index.html>`_.

Please refer to the `documentation <https://cellbender.readthedocs.io/en/latest/>`_ for a quick start tutorial on using CellBender.

Installation and Usage
----------------------

Manual installation
~~~~~~~~~~~~~~~~~~~

The recommended installation is as follows. Create a conda environment and activate it:

.. code-block:: bash

   $ conda create -n cellbender python=3.7
   $ source activate cellbender

Install the `pytables <https://www.pytables.org>`_ module:

.. code-block:: bash

   (cellbender) $ conda install -c anaconda pytables

Install `pytorch <https://pytorch.org>`_ (shown below for CPU; if you have a CUDA-ready GPU, please skip
this part and follow `these <https://pytorch.org/get-started/locally/>`_ instructions instead):

.. code-block:: bash

   (cellbender) $ conda install pytorch torchvision -c pytorch

Clone this repository and install CellBender:

.. code-block:: bash

   (cellbender) $ pip install -e CellBender

Using The Official Docker Image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A GPU-enabled docker image is available from the Google Container Registry (GCR) as:

``us.gcr.io/broad-dsde-methods/cellbender:latest``

Terra Users
~~~~~~~~~~~

For `Terra <https://app.terra.bio>`_ users, a `workflow <https://portal.firecloud.org/#methods/cellbender/remove-background/>`_
is available as:

``cellbender/remove-background``


Citing CellBender
-----------------

If you use CellBender in your research (and we hope you will), please consider
citing `our paper on bioRxiv <https://doi.org/10.1101/791699>`_.

Stephen J Fleming, John C Marioni, and Mehrtash Babadi. CellBender remove-background:
a deep generative model for unsupervised removal of background noise from scRNA-seq
datasets. bioRxiv 791699; doi: `https://doi.org/10.1101/791699 <https://doi.org/10.1101/791699>`_
