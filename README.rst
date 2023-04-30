CellBender
==========

.. image:: https://img.shields.io/github/license/broadinstitute/CellBender?color=white
   :target: LICENSE
   :alt: License

.. image:: https://readthedocs.org/projects/cellbender/badge/?version=latest
   :target: https://cellbender.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://img.shields.io/pypi/v/CellBender.svg
   :target: https://pypi.org/project/CellBender
   :alt: PyPI

.. image:: https://static.pepy.tech/personalized-badge/cellbender?period=total&units=international_system&left_color=grey&right_color=blue&left_text=pypi%20downloads
   :target: https://pepy.tech/project/CellBender
   :alt: Downloads

.. image:: https://img.shields.io/github/stars/broadinstitute/CellBender?color=yellow&logoColor=yellow)
   :target: https://github.com/broadinstitute/CellBender/stargazers
   :alt: Stars

.. image:: docs/source/_static/design/logo_250_185.png
   :alt: CellBender Logo

CellBender is a software package for eliminating technical artifacts from
high-throughput single-cell RNA sequencing (scRNA-seq) data.

The current release contains the following modules. More modules will be added in the future:

* ``remove-background``:

  This module removes counts due to ambient RNA molecules and random barcode swapping from (raw)
  UMI-based scRNA-seq count matrices.  Also works for snRNA-seq and CITE-seq.

Please refer to `the documentation <https://cellbender.readthedocs.io/en/latest/>`_ for a quick start tutorial.

Installation and Usage
----------------------

CellBender can be installed via

.. code-block:: console

  $ pip install cellbender

(and we recommend installing in its own ``conda`` environment, using python 3.8,
to prevent conflicts with other software).

CellBender is run as a command-line tool, as in

.. code-block:: console

  (cellbender) $ cellbender remove-background \
        --cuda \
        --input my_raw_count_matrix_file.h5 \
        --output my_cellbender_output_file.h5

See `the usage documentation <https://cellbender.readthedocs.io/en/latest/usage/index.html>`_
for details.


Using The Official Docker Image
-------------------------------

A GPU-enabled docker image is available from the Google Container Registry (GCR) as:

``us.gcr.io/broad-dsde-methods/cellbender:latest``

Available image tags track release tags in GitHub, and include ``latest``,
``0.1.0``, ``0.2.0``, ``0.2.1``, ``0.2.2``, and ``0.3.0``.


WDL Users
---------

A workflow written in the
`workflow description language (WDL) <https://github.com/openwdl/wdl>`_
is available for CellBender remove-background.

For `Terra <https://app.terra.bio>`_ users, a workflow called
``cellbender/remove-background`` is
`available from the Broad Methods repository
<https://portal.firecloud.org/#methods/cellbender/remove-background/>`_.

There is also a `version available on Dockstore
<https://dockstore.org/workflows/github.com/broadinstitute/CellBender>`_.


Advanced installation
---------------------

From source for development
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a conda environment and activate it:

.. code-block:: console

  $ conda create -n cellbender python=3.8
  $ conda activate cellbender

Install the `pytables <https://www.pytables.org>`_ module:

.. code-block:: console

  (cellbender) $ conda install -c anaconda pytables

Install `pytorch <https://pytorch.org>`_ via
`these instructions <https://pytorch.org/get-started/locally/>`_, for example:

.. code-block:: console

   (cellbender) $ pip install torch

and ensure that your installation is appropriate for your hardware (i.e. that
the relevant CUDA drivers get installed and that ``torch.cuda.is_available()``
returns ``True`` if you have a GPU available.

Clone this repository and install CellBender (in editable ``-e`` mode):

.. code-block:: console

   (cellbender) $ git clone https://github.com/broadinstitute/CellBender.git
   (cellbender) $ pip install -e CellBender


From a specific commit
~~~~~~~~~~~~~~~~~~~~~~

This can be achieved via

.. code-block:: console

   (cellbender) $ pip install --no-cache-dir -U git+https://github.com/broadinstitute/CellBender.git@<SHA>

where ``<SHA>`` must be replaced by any reference to a particular git commit,
such as a tag, a branch name, or a commit sha.


Citing CellBender
-----------------

If you use CellBender in your research (and we hope you will), please consider
citing our paper in Nature Methods:

Stephen J Fleming, Mark D Chaffin, Alessandro Arduini, Amer-Denis Akkad,
Eric Banks, John C Marioni, Anthony A Phillipakis, Patrick T Ellinor,
and Mehrtash Babadi. Unsupervised removal of systematic background noise from
droplet-based single-cell experiments using CellBender.
`Nature Methods` (in press), 2023.

See also `our preprint on bioRxiv <https://doi.org/10.1101/791699>`_.
