<img src="https://github.com/broadinstitute/CellBender/blob/master/docs/source/_static/design/logo_250_185.png"
     alt="CellBender logo"
     style="float: left; margin-right: 10px;" />

# CellBender

CellBender is a software package for eliminating technical artifacts from
high-throughput single-cell RNA sequencing (scRNA-seq) data.

The current release contains the following modules. More modules will be added in the future:

* ``remove-background``:

  This module removes counts due to ambient RNA molecules and random barcode swapping from (raw)
  UMI-based scRNA-seq count matrices. At the moment, only the count matrices produced by the
  CellRanger ``count`` pipeline is supported. Support for additional tools and protocols will be
  added in the future. A quick start tutorial can be found [here](https://github.com/broadinstitute/CellBender/tree/master/examples/remove_background).

Please refer to the documentation for a quick start tutorial on using CellBender.

## Installation and Usage

### Manual installation

The recommended installation is as follows. Create a conda environment and activate it:

```bash
$ conda create -n cellbender python=3.6
$ source activate cellbender
```

Install the [pytables](https://www.pytables.org) module:

```bash
(cellbender) $ conda install -c anaconda pytables
```

Install [pytorch](https://pytorch.org) (shown below for CPU; if you have a CUDA-ready GPU, please skip
this part and follow [these](https://pytorch.org/get-started/locally/) instructions instead):

```bash
(cellbender) $ conda install pytorch torchvision -c pytorch
```

Clone this repository and install CellBender:

```bash
(cellbender) $ pip install -e CellBender
```

### Using The Official Docker Image

A GPU-enabled docker image is available from the Google Container Registry (GCR) as:

```
us.gcr.io/broad-dsde-methods/cellbender:latest
```

### Terra Users

For [Terra](https://app.terra.bio) users, a [workflow](https://portal.firecloud.org/#methods/broad-dsde-methods/cellbender/10/wdl)
is available as:

```
broad-dsde-methods/cellbender
```

### Citing CellBender

If you use CellBender in your research (and we hope you will), please consider
citing our paper on bioRxiv (the link will appear hear soon).
