<img src="https://github.com/broadinstitute/CellBender/blob/master/docs/source/_static/design/logo_250_185.png"
     alt="CellBender logo"
     style="float: left; margin-right: 10px;" />

### CellBender

CellBender is a software package for eliminating technical artifacts from
high-throughput single-cell RNA sequencing (scRNA-seq) data.

The current release contains the following modules. More modules will be added in the future:

* ``remove-background``

  This module removes counts due to ambient RNA molecules and random barcode swapping from (raw)
  UMI-based scRNA-seq count matrices. At the moment, only the count matrices produced by the
  CellRanger ``count`` pipeline is supported. Support for additional tools and protocols will be
  added in the future.

Please refer to the documentation for a quick start tutorial on using CellBender.


