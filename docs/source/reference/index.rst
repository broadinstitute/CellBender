.. _reference:

Reference
=========

.. _remove background reference:

remove-background
-----------------

Command Line Options
~~~~~~~~~~~~~~~~~~~~

.. argparse::
   :module: cellbender.base_cli
   :func: get_populated_argparser
   :prog: cellbender
   :path: remove-background

WDL Workflow Options
~~~~~~~~~~~~~~~~~~~~

A `WDL <https://github.com/openwdl/wdl>`_ script is available as `cellbender_remove_background.wdl
<https://github.com/broadinstitute/CellBender/tree/master/wdl/cellbender_remove_background.wdl>`_,
and it can be used to run ``cellbender remove-background`` from
`Terra <https://app.terra.bio>`_ or from your own
`Cromwell <https://cromwell.readthedocs.io/en/stable/>`_ instance.  The WDL is designed to
make use of an Nvidia Tesla T4 GPU on Google Cloud architecture.

As of v0.3.0, the WDL uses preemptible instances that are a fraction of the cost,
and uses automatic restarting from checkpoints so that work is not lost.

In addition to the above command line options, the workflow has its own set of
input options, which are described in detail
`here <https://github.com/broadinstitute/CellBender/tree/master/wdl>`_.

.. _h5-file-format:

Output h5 file format
~~~~~~~~~~~~~~~~~~~~~

An h5 output file (this one is from the :ref:`tutorial <remove background tutorial>`)
can be examined in detail using PyTables in python:

.. code-block:: python

   # import PyTables
   import tables

   # open the file and take a look at its contents
   with tables.open_file('tiny_output.h5', 'r') as f:
       print(f)

.. code-block:: console

   /home/jupyter/tiny_output.h5 (File) 'CellBender remove-background output'
   Last modif.: 'Wed May  4 19:46:16 2022'
   Object Tree:
   / (RootGroup) 'CellBender remove-background output'
   /droplet_latents (Group) 'Latent variables per droplet'
   /droplet_latents/background_fraction (CArray(1000,), shuffle, zlib(1)) ''
   /droplet_latents/barcode_indices_for_latents (CArray(1000,), shuffle, zlib(1)) ''
   /droplet_latents/cell_probability (CArray(1000,), shuffle, zlib(1)) ''
   /droplet_latents/cell_size (CArray(1000,), shuffle, zlib(1)) ''
   /droplet_latents/droplet_efficiency (CArray(1000,), shuffle, zlib(1)) ''
   /droplet_latents/gene_expression_encoding (CArray(1000, 100), shuffle, zlib(1)) ''
   /global_latents (Group) 'Global latent variables'
   /global_latents/ambient_expression (Array(100,)) ''
   /global_latents/cell_size_lognormal_std (Array()) ''
   /global_latents/empty_droplet_size_lognormal_loc (Array()) ''
   /global_latents/empty_droplet_size_lognormal_scale (Array()) ''
   /global_latents/posterior_regularization_lambda (Array()) ''
   /global_latents/swapping_fraction_dist_params (Array(2,)) ''
   /global_latents/target_false_positive_rate (Array()) ''
   /matrix (Group) 'Counts after background correction'
   /matrix/barcodes (CArray(50500,), zlib(1)) ''
   /matrix/data (CArray(44477,), shuffle, zlib(1)) ''
   /matrix/indices (CArray(44477,), shuffle, zlib(1)) ''
   /matrix/indptr (CArray(50501,), shuffle, zlib(1)) ''
   /matrix/shape (CArray(2,), shuffle, zlib(1)) ''
   /metadata (Group) 'Metadata'
   /metadata/barcodes_analyzed (Array(1000,)) ''
   /metadata/barcodes_analyzed_inds (Array(1000,)) ''
   /metadata/features_analyzed_inds (Array(100,)) ''
   /metadata/fraction_data_used_for_testing (Array()) ''
   /metadata/test_elbo (Array(30,)) ''
   /metadata/test_epoch (Array(30,)) ''
   /metadata/train_elbo (Array(150,)) ''
   /metadata/train_epoch (Array(150,)) ''
   /matrix/features (Group) 'Genes and other features measured'
   /matrix/features/feature_type (CArray(100,), shuffle, zlib(1)) ''
   /matrix/features/genome (CArray(100,), shuffle, zlib(1)) ''
   /matrix/features/id (CArray(100,), shuffle, zlib(1)) ''
   /matrix/features/name (CArray(100,), shuffle, zlib(1)) ''

Any of these bits of data can be accessed directly with PyTables, or by using
python functions from ``cellbender.remove_background.downstream`` such as
``anndata_from_h5()`` that load the metadata as well as the count matrix into
`AnnData format <https://anndata.readthedocs.io/en/latest/>`_
(compatible with analysis in `scanpy <https://scanpy.readthedocs.io/en/stable/>`_).

The names of variables in the h5 file are meant to explain their contents.
All data in the ``/matrix`` group is formatted as `CellRanger v3 h5 data
<https://support.10xgenomics.com/single-cell-gene-expression/software/pipelines/latest/advanced/h5_matrices>`_.

The other root groups in the h5 file ``[droplet_latents, global_latents, metadata]``
contain detailed information inferred by CellBender as part of its latent
variable model of the data generative process (details in our paper).
The ``/droplet_latents`` are variables that have a value inferred for each
droplet, such as ``cell_probability`` and ``cell_size``.  The ``/global_latents`` are
variables that have a fixed value for the whole experiment, such as
``ambient_expression``, which is the inferred profile of ambient RNA in the
sample (normalized so it sums to one).  Finally, ``/metadata`` includes other things
such as the learning curve (``train_elbo`` and ``train_epoch``) and which
features were analyzed during the run (``features_analyzed_inds``, as
integer indices that index ``/matrix/features``).


.. _loading-outputs:

Loading and using outputs
~~~~~~~~~~~~~~~~~~~~~~~~~

As mentioned in the :ref:`tutorial <downstream-example>`, output h5 files can
be opened natively by ``scanpy``
(`scanpy.read_10x_h5() <https://scanpy.readthedocs.io/en/stable/generated/scanpy.read_10x_h5.html>`_)
and (with some effort) by ``Seurat`` (see :ref:`here <open-in-seurat>`), though
there is extra ``cellbender``-specific information in the output files that
``scanpy`` and ``Seurat`` will not load automatically.

Additional loading functionality is distrubuted as part of the ``cellbender``
package and is described below.

Suggested use cases:

- Load just the CellBender output: ``anndata_from_h5()``
- Load the raw input together with the CellBender output (this is great for
  downstream analysis, because it allows you to plot the effect of ``cellbender``,
  and you can always look back at the raw data): ``load_anndata_from_input_and_output()``
- Power users who want to run ``cellbender`` at multiple FPR values and make
  comparisons: ``load_anndata_from_input_and_outputs()``

Example:

.. code-block:: python

    from cellbender.remove_background.downstream import load_anndata_from_input_and_output

    adata = load_anndata_from_input_and_output(
        input_file='my_raw_10x_file.h5',
        output_file='my_cellbender_output_file.h5',
        input_layer_key='raw',  # this will be the raw data layer
    )
    adata

.. code-block:: console

    AnnData object with n_obs × n_vars = 6983 × 37468
    obs: 'background_fraction', 'cell_probability', 'cell_size', 'droplet_efficiency',
        'n_raw', 'n_cellbender'
    var: 'ambient_expression', 'feature_type', 'genome', 'gene_id', 'cellbender_analyzed',
        'n_raw', 'n_cellbender'
    uns: 'cell_size_lognormal_std', 'empty_droplet_size_lognormal_loc',
        'empty_droplet_size_lognormal_scale', 'swapping_fraction_dist_params',
        'target_false_positive_rate', 'features_analyzed_inds',
        'fraction_data_used_for_testing', 'test_elbo', 'test_epoch',
        'train_elbo', 'train_epoch'
    obsm: 'cellbender_embedding'
    layers: 'raw', 'cellbender'

.. automodule:: cellbender.remove_background.downstream
   :members:

The posterior
~~~~~~~~~~~~~

There is an additional output file called ``posterior.h5`` which contains the
full posterior. The structure of this data is a sparse matrix whose shape is
[count matrix entries, number of potential noise counts]. Typically the number
of potential noise counts in any entry of the count matrix is truncated at 50.
The values are log probabilities of the specified number of noise counts in the
given count matrix entry.

The information contained in the posterior can be used to
quantitatively answer questions such as "What is the probability that the
number of viral gene counts in this cell is nonzero?" For help with these kinds
of computations, please open a
`github issue <https://github.com/broadinstitute/CellBender/issues>`_, or see 
the `semi-worked example on the github issue here <https://github.com/broadinstitute/CellBender/issues/299>`_.
