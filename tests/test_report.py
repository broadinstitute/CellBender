"""Tests for HTML report generation."""

import json
import logging
import subprocess
import sys

import pytest

from cellbender.remove_background.report import (
    _log_notebook_cell_errors,
    _run_nbconvert,
    run_notebook_cmd,
    to_html_cmd,
)


@pytest.mark.parametrize("builder", [run_notebook_cmd, to_html_cmd], ids=lambda f: f.__name__)
def test_nbconvert_runs_in_the_current_interpreter(builder):
    """The kernel follows the interpreter nbconvert is launched from.

    Calling the `jupyter` executable found on PATH can pick a different
    environment, whose kernel cannot import cellbender, which produced an empty
    report rather than an error.
    """
    cmd = builder(file="notebook.ipynb", output="out")
    assert cmd[:3] == [sys.executable, "-m", "nbconvert"]


@pytest.mark.parametrize("builder", [run_notebook_cmd, to_html_cmd], ids=lambda f: f.__name__)
def test_nbconvert_output_is_explicit(builder):
    """Do not rely on nbconvert's implicit output naming."""
    cmd = builder(file="notebook.ipynb", output="chosen_name")
    assert "--output" in cmd
    assert cmd[cmd.index("--output") + 1] == "chosen_name"
    assert "notebook.ipynb" in cmd


def test_nbconvert_command_is_a_list_not_a_shell_string():
    """Passed as argv, so paths containing spaces survive."""
    cmd = run_notebook_cmd(file="a notebook.ipynb", output="out")
    assert "a notebook.ipynb" in cmd


def test_run_nbconvert_raises_with_stderr_on_failure():
    """Failures used to be swallowed and surfaced later as a missing file."""
    with pytest.raises(RuntimeError, match="nbconvert failed while testing"):
        _run_nbconvert([sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"], step="testing")


def test_run_nbconvert_error_includes_the_command_and_message():
    try:
        _run_nbconvert([sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"], step="testing")
    except RuntimeError as e:
        assert "boom" in str(e)
        assert "exit code 3" in str(e)
    else:
        pytest.fail("expected RuntimeError")


def test_run_nbconvert_succeeds_quietly():
    _run_nbconvert([sys.executable, "-c", "pass"], step="testing")


def _write_notebook(path, outputs):
    path.write_text(json.dumps({"cells": [{"cell_type": "code", "outputs": outputs}]}))


def test_cell_errors_are_reported(tmp_path, caplog):
    """--allow-errors keeps the report, so failed cells must at least be logged."""
    nb = tmp_path / "executed.ipynb"
    _write_notebook(nb, [{"output_type": "error", "ename": "ModuleNotFoundError", "evalue": "no cellbender"}])
    with caplog.at_level(logging.WARNING, logger="cellbender"):
        _log_notebook_cell_errors(str(nb))
    assert "ModuleNotFoundError" in caplog.text


def test_no_warning_for_a_clean_notebook(tmp_path, caplog):
    nb = tmp_path / "executed.ipynb"
    _write_notebook(nb, [{"output_type": "stream", "text": "fine"}])
    with caplog.at_level(logging.WARNING, logger="cellbender"):
        _log_notebook_cell_errors(str(nb))
    assert caplog.text == ""


def test_unreadable_notebook_does_not_raise(tmp_path):
    """Reporting on errors must never itself break the run."""
    _log_notebook_cell_errors(str(tmp_path / "does_not_exist.ipynb"))
    bad = tmp_path / "bad.ipynb"
    bad.write_text("not json")
    _log_notebook_cell_errors(str(bad))


def test_nbconvert_module_is_importable_in_this_interpreter():
    """The whole fix rests on `python -m nbconvert` working for this python."""
    result = subprocess.run(
        [sys.executable, "-m", "nbconvert", "--version"], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip()
