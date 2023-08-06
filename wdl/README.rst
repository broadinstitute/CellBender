CellBender Workflows
====================

Workflows written in the `workflow description language [WDL]
<https://github.com/openwdl/wdl>`_
enable CellBender commands to be run on cloud computing architecture.

``remove-background``
---------------------

The workflow that runs ``cellbender remove-background`` on a (Tesla T4) GPU on a
Google Cloud virtual machine is located at ``wdl/cellbender_remove_background.wdl``.

Method inputs:
~~~~~~~~~~~~~~

For more information on the input parameters of ``cellbender remove-background``,
refer to the `documentation
<https://cellbender.readthedocs.io/en/latest/help_and_reference/remove_background/index.html>`_.

Required:

* ``input_file_unfiltered``: Path to raw count matrix such as an .h5 file from
  ``cellranger count``, or as a .h5ad or a DGE-format .csv.  In the `Terra <https://app.terra.bio>`_
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
* ``output_bucket_base_directory``: Google bucket path (gsURL) to a directory where
  you want outputs to be copied.  Within that directory, a new folder will appear
  called `sample_name`, and all outputs will go there.  Note that your data will
  then be in two locations in the Cloud: here and wherever Cromwell has its
  execution directory (if using Terra, this is in your workspace's Google bucket).


Optional:

* ``learning_rate``: The learning rate used during inference, as a Float (e.g. ``1e-4``).
* ``epochs``: Number of epochs to train during inference, as an Int (e.g. 150).
* ``low_count_threshold``: An Int that specifies a number of unique UMIs per droplet (e.g. 15).
  Droplets with total unique UMI count below ``low_count_threshold`` will be
  entirely excluded from the analysis.  They are assumed not even to be empty droplets,
  but some barcode error artifacts that are not useful for inference.
* ``projected_ambient_count_threshold``: The larger this number, the fewer genes
  (features) are analyzed, and the faster the tool will run.  Default is 0.1.  The
  value represents the expected number of ambient counts (summed over all cells)
  that we would estimate based on naive assumptions.  Features with fewer expected
  counts than this threshold will be ignored, and the output for those features will
  be identical to the input.

Other input parameters are explained `in the documentation
<https://cellbender.readthedocs.io/en/latest/help_and_reference/remove_background/index.html>`_,
but are only useful in rare cases.

Optional runtime specifications:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Software:

* ``docker_image``: Name of the docker image which will be used to run the
  ``cellbender remove-background`` command, as a String that references an image
  with CellBender installed (e.g. "us.gcr.io/broad-dsde-methods/cellbender:0.3.0").
  Note that this WDL may not be compatible with other versions of CellBender due
  to changes in input arguments.

Hardware:

* ``hardware_gpu_type``: Specify a `GPU type <https://cloud.google.com/compute/docs/gpus>`_
* ``hardware_zones``: Specify `Google Cloud compute zones
  <https://cloud.google.com/compute/docs/regions-zones/>`_ as a whitespace-delimited String.
  The chosen zones should have the appropriate GPU hardware available, otherwise this
  will cause an error.
* ``hardware_disk_size_GB``: Specify the size of the disk attached to the VM, as
  an Int in units of GB.
* ``hardware_preemptible_tries``: Specify the number of preemptible runs to attempt,
  as an Int.  Preemptible runs are 1/3 to 1/4 the cost.  If the run gets preempted
  ``hardware_preemptible_tries`` times, a final non-preemptible run is carried out.
  Work is not lost during preemption because the workflow uses
  `checkpointing <https://cromwell.readthedocs.io/en/stable/optimizations/CheckpointFiles/>`_
  to pick up (near) where it left off.

Outputs:
~~~~~~~~

``cellbender remove-background`` outputs several files, and each of these files is
included as an output of the workflow.

If multiple FPR values are specified, then separate ``.h5`` and ``report.html``
``metrics.csv`` files will be produced, one for each FPR.

* ``h5_array``: Array of output count matrix files (one for each FPR), with
  background RNA removed.
* ``html_report_array``: Array of HTML output reports (one for each FPR)
* ``metrics_csv_array``: Array of CSV files that include output metrics (one for
  each FPR)
* ``output_directory``: If the input `output_base_directory` was blank, this
  will be black too. Same as the workflow input parameter, but with the
  `sample_name` subdirectory added: useful for populating
  a column of the Terra data model.  All outputs are copied here.
* ``cell_barcodes_csv``: CSV file containing all the droplet barcodes which were determined to have
  a > 50% posterior probability of containing cells.  Barcodes are written in plain text.
  This information is also contained in each of the above outputs, but is included as a separate
  output for convenient use in certain downstream applications.
* ``summary_pdf``: PDF file that provides a quick graphical summary of the run.
* ``log``: Log file produced by the ``cellbender remove-background`` run.


Cost:
~~~~~

The cost to run a single sample in v0.3.0 on a preemptible Nvidia Tesla T4 GPU
on Google Cloud hovers somewhere in the ballpark of
