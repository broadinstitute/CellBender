"""Tests for the outputs the pipeline produces itself, and their dependencies."""

import subprocess
import sys

import pytest

from cellbender.remove_background.summary import (
    REPORT_EXTRA_HINT,
    _check_jupyter_available,
    run_notebook_str,
    to_html_str,
)


def test_pipeline_import_does_not_pull_in_ipython():
    """The base install must not need the optional report dependencies.

    report.py runs inside the Jupyter notebook and imports IPython, so run.py
    must reach the summary PDF and the notebook driver without importing it.
    Run in a subprocess so that other tests having imported IPython cannot mask
    a regression here.
    """
    code = (
        "import sys\n"
        "import cellbender.remove_background.run\n"
        "leaked = sorted(m for m in sys.modules if m.split('.')[0] in "
        "{'IPython', 'nbconvert', 'ipykernel'})\n"
        "print(','.join(leaked))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    leaked = result.stdout.strip()
    assert leaked == "", f"importing the pipeline pulled in report-only dependencies: {leaked}"


def test_check_jupyter_available_message_points_at_the_extra(monkeypatch):
    monkeypatch.setattr("cellbender.remove_background.summary.shutil.which", lambda _: None)
    with pytest.raises(RuntimeError, match=r"pip install cellbender\[report\]"):
        _check_jupyter_available()
    assert "cellbender[report]" in REPORT_EXTRA_HINT


def test_check_jupyter_available_passes_when_present(monkeypatch):
    monkeypatch.setattr("cellbender.remove_background.summary.shutil.which", lambda _: "/usr/bin/jupyter")
    _check_jupyter_available()


def test_notebook_commands_reference_the_given_file():
    assert "some_notebook.ipynb" in run_notebook_str(file="some_notebook.ipynb")
    assert "some_notebook.ipynb" in to_html_str(file="some_notebook.ipynb", output="out.html")
