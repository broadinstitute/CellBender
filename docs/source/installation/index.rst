.. _installation:

Installation
============

Via pip
-------

Python packages can be conveniently installed from the Python Package Index (PyPI)
using `pip install <https://pip.pypa.io/en/stable/cli/pip_install/>`_.
CellBender is `available on PyPI <https://pypi.org/project/cellbender/>`_
and can be installed via

.. code-block:: console

  $ pip install cellbender

If your machine has a GPU with appropriate drivers installed, it should be
automatically detected, and the appropriate version of PyTorch with CUDA support
should automatically be downloaded as a CellBender dependency.

We recommend installing CellBender in its own
`conda environment <https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html>`_.
This allows for easier installation and prevents conflicts with any other python
packages you may have installed.

.. code-block:: console

  $ conda create -n cellbender python=3.7
  $ conda activate cellbender
  (cellbender) $ pip install cellbender


Installation from source
------------------------

Create a conda environment and activate it:

.. code-block:: console

  $ conda create -n cellbender python=3.7
  $ conda activate cellbender

Install the `pytables <https://www.pytables.org>`_ module:

.. code-block:: console

  (cellbender) $ conda install -c anaconda pytables

Install `pytorch <https://pytorch.org>`_ via
`these instructions <https://pytorch.org/get-started/locally/>`_:

.. code-block:: console

   (cellbender) $ pip install torch

and ensure that your installation is appropriate for your hardware (i.e. that
the relevant CUDA drivers get installed and that ``torch.cuda.is_available()``
returns ``True`` if you have a GPU available.

Clone this repository and install CellBender (in editable ``-e`` mode):

.. code-block:: console

   (cellbender) $ git clone https://github.com/broadinstitute/CellBender.git
   (cellbender) $ pip install -e CellBender


Install a specific commit directly from GitHub
----------------------------------------------

This can be achieved via

.. code-block:: console

   (cellbender) $ pip install --no-cache-dir -U git+https://github.com/broadinstitute/CellBender.git@<SHA>

where ``<SHA>`` must be replaced by any reference to a particular git commit,
such as a tag, a branch name, or a commit sha.

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
