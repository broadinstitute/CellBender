"""Outputs the pipeline itself produces: the summary PDF and the HTML report driver.

Kept separate from report.py because report.py is imported and executed inside
the Jupyter notebook and therefore depends on IPython. This module is imported
by run.py on every run, so it must stay importable with only the base
dependencies installed.
"""

import logging
import os
import shutil
import subprocess
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch

from cellbender.remove_background import consts

logger = logging.getLogger("cellbender")
TIMEOUT = 1200  # twenty minutes should always be way more than enough

REPORT_EXTRA_HINT = (
    "The HTML report needs the optional report dependencies. Install them with:\n    pip install cellbender[report]"
)


def _check_jupyter_available() -> None:
    """Fail with an actionable message rather than an opaque subprocess error."""
    if shutil.which("jupyter") is None:
        raise RuntimeError(f"Cannot find the 'jupyter' executable on PATH. {REPORT_EXTRA_HINT}")


def run_notebook_str(file):
    return f"jupyter nbconvert --ExecutePreprocessor.timeout={TIMEOUT} --to notebook --allow-errors --execute {file}"


def to_html_str(file, output):
    return f"jupyter nbconvert --to html --TemplateExporter.exclude_input=True {file}"


def _run_notebook(file):
    shutil.copy(file, "tmp.report.ipynb")
    subprocess.run(run_notebook_str(file="tmp.report.ipynb"), shell=True)
    os.remove("tmp.report.ipynb")
    return "tmp.report.nbconvert.ipynb"


def _to_html(file, output) -> str:
    subprocess.run(to_html_str(file=file, output=output), shell=True)
    shutil.move(file.replace(".ipynb", ".html"), output)
    os.remove(file)
    return output


def _postprocess_html(file: str, title: str):
    try:
        with open(file, mode="r", encoding="utf8", errors="surrogateescape") as f:
            html = f.read()
        html = html.replace("<title>tmp.report.nbconvert</title>", f"<title>{title}</title>")
        with open(file, mode="w", encoding="utf8", errors="surrogateescape") as f:
            f.write(html)
    except Exception:
        logger.warning(
            "Failed to overwrite default HTML report title. This is purely aesthetic and does not affect output."
        )


def run_notebook_make_html(file, output) -> str:
    """Run Jupyter notebook to populate report and then convert to HTML.

    Args:
        file: Notebook file
        output: Output file.  Should end in ".html"

    Returns:
        output: Output file

    """
    assert output.endswith(".html"), "Output HTML filename should end with .html"
    _check_jupyter_available()
    html_file = _to_html(file=_run_notebook(file), output=output)
    _postprocess_html(
        file=html_file,
        title=("CellBender: " + os.path.basename(output).replace("_report.html", "")),
    )
    return html_file


def pca_2d(mat: np.ndarray) -> torch.Tensor:
    """Perform PCA using pytorch and return top 2 PCs

    Args:
        mat: matrix where rows are observations and columns are features

    Returns:
        out: matrix where rows are observations and columns are top 2 PCs
    """

    A = torch.as_tensor(mat).float()
    U, S, V = torch.pca_lowrank(A)
    return torch.matmul(A, V[:, :2])


def plot_summary(loss: Dict[str, Dict[str, Any]], umi_counts: np.ndarray, p: np.ndarray, z: np.ndarray):
    """Output summary plot with three panels: training, cells, latent z."""

    fig = plt.figure(figsize=(6, 18))

    # Plot the train error.
    plt.subplot(3, 1, 1)
    try:
        plt.plot(loss["train"]["elbo"], ".--", label="Train")

        # Plot the test error, if there was held-out test data.
        if "test" in loss.keys():
            if len(loss["test"]["epoch"]) > 0:
                plt.plot(loss["test"]["epoch"], loss["test"]["elbo"], "o:", label="Test")
                plt.legend()

        ylim_low = max(loss["train"]["elbo"][0], loss["train"]["elbo"][-1] - 2000)
        try:
            ylim_high = max(max(loss["train"]["elbo"]), max(loss["test"]["elbo"]))
        except ValueError:
            ylim_high = max(loss["train"]["elbo"])
        ylim_high = ylim_high + (ylim_high - ylim_low) / 20
        plt.gca().set_ylim((ylim_low, ylim_high))
    except Exception:
        pass

    plt.xlabel("Epoch")
    plt.ylabel("ELBO")
    plt.title("Progress of the training procedure")

    # Plot the barcodes used, along with the inferred
    # cell probabilities.
    plt.subplot(3, 1, 2)
    count_order = np.argsort(umi_counts)[::-1]
    plt.semilogy(umi_counts[count_order], color="black")
    plt.ylabel("UMI counts")
    plt.xlabel("Barcode index, sorted by UMI count")
    if p is not None:  # The case of a simple model.
        plt.gca().twinx()
        plt.plot(p[count_order], ".:", color="red", alpha=0.3, rasterized=True)
        plt.ylabel("Cell probability", color="red")
        plt.ylim([-0.05, 1.05])  # type: ignore [call-overload]
        plt.title("Determination of which barcodes contain cells")
    else:
        plt.title("The subset of barcodes used for training")

    plt.subplot(3, 1, 3)
    if p is None:
        p = np.ones(z.shape[0])

    # Do PCA on the latent encoding z.
    z_pca = pca_2d(z[p >= consts.CELL_PROB_CUTOFF])

    # Plot the latent encoding via PCA.
    plt.plot(z_pca[:, 0], z_pca[:, 1], ".", ms=3, color="black", alpha=0.3, rasterized=True)
    plt.ylabel("PC 1")
    plt.xlabel("PC 0")
    plt.title("PCA of latent encoding of gene expression in cells")

    return fig
