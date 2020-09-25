.. _remove background reference:

remove-background
=================

Command Line Options
--------------------

.. argparse::
   :module: cellbender.base_cli
   :func: get_populated_argparser
   :prog: cellbender
   :path: remove-background

WDL Workflow Options
--------------------

A `WDL <https://github.com/openwdl/wdl>`_ script is available as `cellbender_remove_background.wdl
<https://github.com/broadinstitute/CellBender/tree/master/wdl/cellbender_remove_background.wdl>`_,
and it can be used to run ``cellbender remove-background`` from
`Terra <https://app.terra.bio>`_ or from your own
`Cromwell <https://cromwell.readthedocs.io/en/stable/>`_ instance.  The WDL is designed to
make use of Tesla K80 GPUs on Google Cloud architecture.

In addition to the above command line options, the workflow has its own set of
input options, which are described in detail
`here <https://github.com/broadinstitute/CellBender/tree/master/wdl>`_.

.. _remove background reference troubleshooting:

Troubleshooting
---------------

* The learning curve in the output PDF has large downward spikes, or looks super wobbly.

  * This could indicate instabilities during training that should be addressed. The solution
    is typically to reduce the ``--learning-rate`` by a factor of two.

* The following warning is emitted in the log file: "Warning: few empty droplets identified.
  Low UMI cutoff may be too high. Check the UMI decay curve, and decrease the
  ``--low-count-threshold`` parameter if necessary."

  * This warning indicates that no "surely empty" droplets were identified in the analysis.
    This means that the "empty droplet plateau" could not be identified.  The most likely
    explanation is that the level of background RNA is extremely low, and that the value
    of ``--low-count-threshold`` exceeds this level.  This would result in the empty
    droplet plateau being excluded from the analysis, which is not advisable.  This can be
    corrected by decreasing ``--low-count-threshold`` from its default of 15 to a value like 5.


* There are too many cells called.

  * Are there?  ``remove-background`` equates "cell probability" with "the probability that
    a given droplet is not empty."  These non-empty droplets might not all contain healthy
    cells with high counts.  Nevertheless, the posterior probability that they are not empty
    is greater than 0.5.  The recommended procedure
    would be to filter cells based on other criteria downstream.  Certainly filter for percent
    mitochondrial reads.  Potentially filter for number of genes expressed as well, if
    this does not lead to complete loss of a low-expressing cell type.
  * Experiment with increasing ``--total-droplets-included``.
  * Experiment with increasing or decreasing ``--empty-drop-training-fraction``.
  * As a last resort, try decreasing ``--expected-cells`` by quite a bit.


* There are too few cells called.

  * Try estimating ``--expected-cells`` from the UMI curve rather than a priori, and
    increase the number if necessary.
  * Experiment with increasing or decreasing ``--total-droplets-included``.


* The PCA plot of latent gene expression shows no clusters or structure.

  * Has training converged?  Training should proceed for at least 150 epochs.  Check to
    make sure that the ELBO has nearly reached a plateau, indicating that training is
    complete.  Try increasing ``--epochs`` to 300 and see if the plot changes.
  * This is not necessarily a bad thing, although it indicates that cells in the experiment
    had a continuum of expression, or that there was only one cell type.  If this is
    known to be false, some sort of QC failure with the experiment would be suspected.
    Perform a downstream clustering analysis with and without ``cellbender remove-background``
    and compare the two.
