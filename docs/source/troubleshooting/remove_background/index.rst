.. _remove background troubleshooting:

remove-background
-----------------

Feel free to check the
`issues on github <https://github.com/broadinstitute/CellBender/issues?q=is%3Aissue>`_,
where there are several answered questions
you can search through.  If you don't see an answer there or below, please consider
opening a new issue on github to ask your question.

FAQ
~~~

* :ref:`What are the "best practices" for running the tool? <a1>`

* :ref:`What are the "best practices" for incorporating CellBender into my analysis pipeline? <a2>`

* :ref:`How do I load and use the output? <a22>`

* :ref:`What are the details of the output h5 file format? <a23>`

* :ref:`Do I really need a GPU to run this? <a3>`

* :ref:`How do I determine the right "--fpr"? <a4>`

* :ref:`How do I determine the right "--expected-cells"? <a5>`

* :ref:`How do I determine the right "--total-droplets-included"? <a6>`

* :ref:`Does CellBender work with CITE-seq data?  Or other non-"Gene Expression" features? <a7>`

* :ref:`Could I run CellBender using only "Gene Expression" features and ignore other features? <a8>`

* :ref:`Could I run CellBender using only "Antibody Capture" features and not Gene Expression? <a24>`

* :ref:`Where can I find the ambient RNA profile inferred by CellBender? <a9>`

* :ref:`The code completed, but how do I know if it "worked"? / How do I know when I need to re-run with different parameters? <a10>`

* :ref:`It seems like CellBender called too many cells <a11>`

* :ref:`It seems like CellBender called too few cells <a12>`

* :ref:`The learning curve looks weird. / What is the learning curve supposed to look like? <a13>`

* :ref:`What is the "metrics.csv" output for, and how do I interpret the metrics? <a14>`

* :ref:`My HTML report summary (at the bottom) said there were some warnings.  What should I do about that? <a15>`

* :ref:`The tool failed to produce an HTML report <a16>`

* :ref:`I ran the WDL in Terra and the job "Failed" with PAPI error code 9 <a17>`

* :ref:`How much does it cost, per sample, to run WDL in Terra? <a18>`

* :ref:`I am getting a GPU-out-of-memory error (process "Killed") <a19>`

* :ref:`I got a "nan" error and the tool crashed <a20>`

* :ref:`There was an error! <a21>`


Answers / Troubleshooting Tips
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _a1:

* What are the "best practices" for running the tool?

  * See the :ref:`recommended best practices <best-practices>`

.. _a2:

* What are the "best practices" for incorporating CellBender into my analysis pipeline?

  * See the :ref:`proposed pipeline <proposed-pipeline>`

.. _a22:

* How do I load and use the output?

  * The data can easily be loaded as an ``AnnData`` object in python, to be used in
    ``scanpy``, or it can also be loaded by Seurat, see the
    :ref:`example here <downstream-example>`
  * We recommend loading the input and output data (together) using the function
    ``cellbender.remove_background.downstream.load_anndata_from_input_and_output()``
    from the CellBender package, since that will create an ``AnnData`` with
    separate layers for the raw data and the CellBender output data.  This is
    quite handy for downstream work.  There are several simple data-loading
    functions in ``cellbender.remove_background.downstream`` that might be
    useful.

  .. code-block:: python

     from cellbender.remove_background.downstream import load_anndata_from_input_and_output

     adata = load_anndata_from_input_and_output(input_file='tiny_raw_feature_bc_matrix.h5ad',
                                                output_file='tiny_output.h5')
     adata

  .. code-block:: console

     AnnData object with n_obs × n_vars = 1000 × 100
         obs: 'background_fraction', 'cell_probability', 'cell_size', 'droplet_efficiency', 'n_cellranger', 'n_cellbender'
         var: 'ambient_expression', 'features_analyzed_inds', 'feature_type', 'genome', 'gene_id', 'n_cellranger', 'n_cellbender'
         uns: 'cell_size_lognormal_std', 'empty_droplet_size_lognormal_loc', 'empty_droplet_size_lognormal_scale', 'posterior_regularization_lambda', 'swapping_fraction_dist_params', 'target_false_positive_rate', 'fraction_data_used_for_testing', 'test_elbo', 'test_epoch', 'train_elbo', 'train_epoch'
         obsm: 'cellbender_embedding'
         layers: 'cellranger', 'cellbender'

.. _a23:

* What are the details of the output h5 file format?

  * :ref:`See here <h5-file-format>`

.. _a3:

* Do I really need a GPU to run this?

  * It's not absolutely necessary, but the code takes a long time to run  on a full
    dataset on a CPU.
  * While running on a GPU might seem like an insurmountable obstacle for those without
    a GPU handy, consider trying out our
    `workflow on Terra <https://portal.firecloud.org/#methods/cellbender/remove-background/>`_,
    which runs on a GPU on Google Cloud at the click of a button.
  * Others have successfully run on Google Colab notebooks for free.  Since CellBender
    produces checkpoint files during training (``ckpt.tar.gz``), you can even pick up
    where you left off if you get preempted.
  * If you really want to use a CPU only, then consider things that will speed up the
    run, like using fewer ``--total-droplets-included``, and increasing the threshold
    ``--projected-ambient-count-threshold`` so that fewer features are analyzed,
    and maybe decreasing ``--empty-drop-training-fraction``, so that each minibatch
    has fewer empty droplets.

.. _a4:

* How do I determine the right ``--fpr``?

  * See the :ref:`recommended best practices <best-practices>`

.. _a5:

* How do I determine the right ``--expected-cells``?

  * See the :ref:`recommended best practices <best-practices>`

.. _a6:

* How do I determine the right ``--total-droplets-included``?

  * See the :ref:`recommended best practices <best-practices>`

.. _a7:

* Does CellBender work with CITE-seq data?  Or other non-``Gene Expression`` features?

  * Absolutely, `as shown here <https://github.com/broadinstitute/CellBender/issues/114>`_
    and in our paper.  The results for ``Antibody Capture`` data look even better than
    for gene expression, due to the higher ambient background for that modality.

.. _a8:

* Could I run CellBender using only ``Gene Expression`` features and ignore other features?

  * If you want to, you can (though it works great with ``Antibody Capture`` data):
    just use the ``--exclude-feature-types`` input parameter.

.. _a24:

* Could I run CellBender using only ``Antibody Capture`` features and not ``Gene Expression``?

  * No, it is not a good idea to exclude ``Gene Expression`` features for the following
    reason: the ``Gene Expression`` features form the basis of a good prior on "cell type",
    which the method heavily relies on to denoise. Other features, without ``Gene Expression``,
    are probably too sparse to cluster similar cells together and form a good prior for
    "which cells are similar to which others".

.. _a9:

* Where can I find the ambient RNA profile inferred by CellBender?

.. _a10:

* The code completed, but how do I know if it "worked"?  / How do I know when I
  need to re-run with different parameters?

.. _a11:

* It seems like CellBender called too many cells

.. _a12:

* It seems like CellBender called too few cells

.. _a13:

* The learning curve looks weird. / What is the learning curve supposed to
  look like?

.. _a14:

* What is the ``metrics.csv`` output for, and how do I interpret the metrics?

.. _a15:

* My HTML report summary (at the bottom) said there were some warnings.  What
  should I do about that?

.. _a16:

* The tool failed to produce an HTML report

.. _a17:

* I ran the WDL in Terra and the job ``Failed`` with PAPI error code 9

.. _a18:

* How much does it cost, per sample, to run WDL in Terra?

.. _a19:

* I am getting a GPU-out-of-memory error (process ``Killed``)

.. _a20:

* I got a ``nan`` error and the tool crashed

.. _a21:

* There was an error!







(Most of these points are answers which are linked to from the FAQ above.)

* The ambient RNA profile lives at ``/global_latents/ambient_profile`` in the
  output h5 file tree.  Don't want to dig around in h5 files?  Don't worry...
  just use the AnnData loader function ``cellbender.remove_background.downstream.anndata_from_h5()``
  to load an AnnData object which will have the ambient RNA profile accessible
  as ``adata.var['ambient_expression']``

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
