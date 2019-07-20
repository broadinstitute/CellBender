.. _introduction:

What is CellBender?
=====================

CellBender is a software package for eliminating technical artifacts from high-throughput single-cell RNA sequencing
(scRNA-seq) data.

Scope and Purpose
-----------------

Despite the recent progress in improving, optimizing and standardizing scRNA-seq protocols, the complexity of
scRNA-seq experiments leaves room for systematic biases and background noise in the raw observations. These nuisances
can be traced back to undesirable enzymatic processes that produce spurious library fragments, contamination by
exogeneous or endogenous ambient transcripts, impurity of barcode beads, and barcode swapping during amplification
and/or sequencing.

The main purpose of CellBender is to take raw gene-by-cell count matrices and molecule-level information produced
by 3rd party pipelines (e.g. CellRanger, Alevin), to model and remove systematic biases and background noise, and
to produce improved estimates of gene expression.

As such, CellBender relies on an external tool for primary processing of the raw data obtained from the
sequencer (e.g. BCL or FASTQ files). These basic processing steps lie outside of the scope of CellBender
and include (pseudo-)alignment and annotation of reads, barcode error correction, and generation of raw gene-by-cell
count matrices. Upcoming modules of CellBender will further utilize molecule-level information (e.g. observed reads
per molecule, transcript equivalent classes, etc.).

Modules
-------

The current version of CellBender contains the following modules. More modules will be added in the future:

* ``remove-background``: This module removes counts due to ambient RNA molecules and random barcode swapping from
  (raw) UMI-based scRNA-seq gene-by-cell count matrices. At the moment, only the count matrices produced by the
  CellRanger `count` pipeline is supported. Support for additional tools and protocols will be added in the future.
  A quick-start tutorial can be found :ref:`here <remove background tutorial>`.
