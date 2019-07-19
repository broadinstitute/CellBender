.. _introduction:

What is *CellBender*?
=====================

*CellBender* is a software package for eliminating technical artifacts from high-throughput single-cell RNA sequencing
(scRNA-seq) data.


*TODO*: The scope of CellBender (what it does, what it doesn't).


The current release contains the following modules. More modules will be added in the future:

* ``remove-background``: This module removes counts due to ambient RNA molecules
  and random barcode swapping from (raw) UMI-based scRNA-seq count matrices.
  At the moment, only the count matrices produced by the CellRanger count pipeline is
  supported. Support for additional tools and protocols will be added in the future.

