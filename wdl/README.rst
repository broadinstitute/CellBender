CellBender Workflows
====================

Workflows written in the `workflow description language [WDL]
<https://github.com/openwdl/wdl>`_
enable CellBender commands to be run on cloud computing architecture.

``remove-background``
---------------------

The workflow that runs ``cellbender remove-background`` on a (Tesla K80) GPU on a
Google Cloud virtual machine is located at ``wdl/cellbender_remove_background.wdl``.

Method inputs:
~~~~~~~~~~~~~~

For more information on the input parameters of ``cellbender remove-background``,
refer to the `documentation
<https://cellbender.readthedocs.io/en/latest/help_and_reference/remove_background/index.html>`_.

Required:

* ``input_10x_h5_file_or_mtx_directory``: Path to raw count matrix output by 10x
  ``cellranger count`` as either a .h5 file or as the directory that contains
  ``matrix.mtx(.gz)``.  In the `Terra <https://app.terra.bio>`_
  data model, this could be ``this.h5_file``, where "h5_file" is the column of
  the data model that contains the path to the raw count matrix h5.  Alternatively,
  this could be a Google bucket path as a String (e.g.
  "gs://my_bucket_name/cellranger_count_output_path/raw_gene_bc_matrix_h5.h5").
* ``sample_name``: Name of sample, used as the sample identifier (a String).
  In the `Terra <https://app.terra.bio>`_ data model, this should be ``this.sample_id``.

Optional and recommended:

* ``fpr``: Float value(s) in the range (0, 1) that specify "how much noise to remove". Each
  value will produce a separate output count matrix. FPR stands for the expected false
  positive rate, where a false positive is a real count that is erroneously removed.
  A value of 0.01 means "remove as much noise as possible while not removing more than
  1% of the real signal". (A value of 0 would return the input, and a value of 1 would
  return a completely empty count matrix.)
* ``expected_cells``: Number of cells expected a priori based on experimental
  design, as an Int (e.g. 5000).
* ``total_droplets_included``: Total number of droplets to include in the analysis,
  as an Int (e.g. 25000).  The ``total_droplets_included`` largest UMI-count droplets will
  be called as either cell or empty droplet by the inference procedure.  Any
  droplets not included in the ``total_droplets_included`` largest UMI-count
  droplets will be treated as surely empty.


Optional:

* ``learning_rate``: The learning rate used during inference, as a Float (e.g. ``1e-4``).
* ``epochs``: Number of epochs to train during inference, as an Int (e.g. 150).
* ``model``: One of {"full", "ambient", "swapping", "simple"}.  This specifies how
  the count data should be modeled.  "full" specifies an ambient RNA plus chimera
  formation model, while "ambient" specifies a model with only ambient RNA, and
  "swapping" specifies a model with only chimera formation.  "simple" should not
  be used in this context.
* ``low_count_threshold``: An Int that specifies a number of unique UMIs per droplet (e.g. 15).
  Droplets with total unique UMI count below ``low_count_threshold`` will be
  entirely excluded from the analysis.  They are assumed not even to be empty droplets,
  but some barcode error artifacts that are not useful for inference.
* ``blacklist_genes``: A whitespace-delimited String of integers
  (e.g. "523 10021 10022 33693 33694") that specifies genes that should be completely
  excluded from analysis.  Counts of these genes are set to zero in the output count matrix.
  Genes are specified by the integer that indexes them in the count matrix.

Optional but discouraged:

[There should not be any need to change these parameters from their default values.]

* ``z_dim``: Dimension of the latent gene expression space, as an Int.  Use a smaller
  value (e.g. 20) for slightly more imputation, and a larger value (e.g. 200) for
  less imputation.
* ``z_layers``: Architecture of the neural network autoencoder for the latent representation
  of gene expression.  ``z_layers`` specifies the size of each hidden layer.
  Input as a whitespace-delimited String of integers, (e.g. "1000").
  Only use one hidden layer.  [Two hidden layers could be specified with "500 100" for
  example, but only one hidden layer should be used.]
* ``empty_drop_training_fraction``: Specifies what fraction of the data in each
  minibatch should come from surely empty droplets, as a Float (e.g. 0.3).

Optional runtime specifications:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Software:

* ``docker_image``: Name of the docker image which will be used to run the
  ``cellbender remove-background`` command, as a String that references an image
  either on the Google Container Registry (e.g. "us.gcr.io/broad-dsde-methods/cellbender:latest")
  or Dockerhub (e.g. "dockerhub_username/image_name").  Images should be 3GB or smaller.

Hardware:

* ``hardware_gpu_type``: Specify a `GPU type <https://cloud.google.com/compute/docs/gpus>`_
* ``hardware_zones``: Specify `Google Cloud compute zones
  <https://cloud.google.com/compute/docs/regions-zones/>`_ as a whitespace-delimited String.
  The chosen zones should have the appropriate GPU hardware available, otherwise this
  will cause an error.
* ``hardware_disk_size_GB``: Specify the size of the disk attached to the VM, as
  an Int in units of GB.
* ``hardware_preemptible_tries``: Specify the number of preemptible runs to attempt,
  as an Int.  Preemptible runs are 1/3 to 1/4 the cost.  If the run gets pre-empted
  ``hardware_preemptible_tries`` times, a final non-preemptible run is carried out.

Outputs:
~~~~~~~~

``cellbender remove-background`` outputs five files, and each of these output files is
included as an output of the workflow.

If multiple FPR values are specified, then separate ``h5`` and

* ``h5_array``: Array of output count matrix files, with background RNA removed.
* ``output_directory``: Same as the workflow input parameter, useful for populating
  a column of the Terra data model.
* ``csv``: CSV file containing all the droplet barcodes which were determined to have
  a > 50% posterior probability of containing cells.  Barcodes are written in plain text.
  This information is also contained in each of the above outputs, but is included as a separate
  output for convenient use in certain downstream applications.
* ``pdf``: PDF file that provides a standard graphical summary of the inference procedure.
* ``log``: Log file produced by the ``cellbender remove-background`` run.
