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
