"""Handle the saving and loading of models from checkpoint files."""

import argparse
import concurrent.futures
import glob
import hashlib
import io
import json
import logging
import os
import pickle
import random
import shutil
import tarfile
import tempfile
import traceback
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from cellbender.remove_background.model import RemoveBackgroundPyroModel

import dill
import numpy as np
import torch

from cellbender.remove_background import consts
from cellbender.remove_background.data.dataprep import DataLoader

logger = logging.getLogger("cellbender")

USE_PYRO = True
try:
    import pyro
except ImportError:
    USE_PYRO = False
USE_CUDA = torch.cuda.is_available()

_ckpt_executor: concurrent.futures.ThreadPoolExecutor = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="checkpoint"
)
_pending_ckpt: Optional[concurrent.futures.Future] = None


def _get_scalar_param_store_state(model_obj: "RemoveBackgroundPyroModel") -> Dict:
    """Return a Pyro param store state dict containing only the *scalar* (non-module)
    parameters — i.e. everything that is NOT an ``nn.Parameter`` tracked by
    ``model_obj``.

    Encoder / decoder ``nn.Parameter`` tensors are intentionally excluded here
    because they are also saved verbatim in ``_model.state_dict.torch``.
    Loading both copies creates two separate tensor objects with the same
    value but different Python identities, which causes a one-ULP divergence
    when the Pyro optimiser uses one copy while the model uses the other.
    By excluding them from the param store backup, the checkpoint-restore path
    matches the fresh-training path: when ``pyro.module(...,
    update_module_params=True)`` is first called, the model's own (already
    correctly initialised) parameters get registered into the param store
    without any replacement, giving a single shared tensor object.
    """
    ps = pyro.get_param_store()
    # Identify pyro param names that correspond to model nn.Parameters.
    # In the one-shot / round-1 path the param store tensors ARE the model
    # parameter objects (same identity), so we can look them up by id.
    model_param_ids = {id(p) for _, p in model_obj.named_parameters()}
    module_keys = {name for name, p in ps._params.items() if id(p) in model_param_ids}
    full_state = ps.get_state()
    return {
        "params": {k: v for k, v in full_state["params"].items() if k not in module_keys},
        "constraints": {k: v for k, v in full_state["constraints"].items() if k not in module_keys},
    }


def _build_model_meta_dict(model_obj: "RemoveBackgroundPyroModel") -> Optional[Dict]:
    """Build the JSON-serializable metadata dict for a RemoveBackgroundPyroModel.
    Returns None if the model does not have the expected attributes."""
    if not hasattr(model_obj, "empty_UMI_threshold"):
        return None
    umi_thresh = model_obj.empty_UMI_threshold
    return {
        "model_type": model_obj.model_type,
        "z_dim": int(model_obj.z_dim),
        "z_hidden_dims": model_obj.z_hidden_dims or [],
        "n_analyzed_genes": int(model_obj.n_genes),
        "n_droplets": int(model_obj.n_droplets),
        "empty_UMI_threshold": umi_thresh.item() if isinstance(umi_thresh, torch.Tensor) else int(umi_thresh),
        "log_counts_crossover": float(model_obj.log_counts_crossover),
        "loss": model_obj.loss,
    }


def _save_model_meta(model_obj: "RemoveBackgroundPyroModel", path: str) -> None:
    """Write a JSON file with the model config and training history needed to
    reconstruct the model architecture from scratch on checkpoint load.
    No-op if the model lacks RemoveBackgroundPyroModel-specific attributes."""
    meta = _build_model_meta_dict(model_obj)
    if meta is None:
        return
    with open(path, "w") as f:
        json.dump(meta, f)


def load_optim_from_bytes(
    scheduler: pyro.optim.PyroOptim,
    optim_state_bytes: bytes,
    map_location: Optional[torch.device] = None,
) -> None:
    """Restore optimizer state from a raw bytes buffer previously produced by
    :func:`scheduler.save`.  Uses a short-lived temp file since
    :meth:`PyroOptim.load` requires a path string."""
    ntf_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pyro", delete=False) as ntf:
            ntf.write(optim_state_bytes)
            ntf_path = ntf.name
        kwargs: Dict = {} if map_location is None else {"map_location": map_location}
        scheduler.load(ntf_path, **kwargs)
    finally:
        if ntf_path is not None:
            try:
                os.unlink(ntf_path)
            except OSError:
                pass


def save_random_state(filebase: str) -> List[str]:
    """Write states of various random number generators to files.

    NOTE: the pyro.util.get_rng_state() is a compilation of python, numpy, and
        torch random states.  Here, we save the three explicitly ourselves.
        This is useful for potential future usages outside of pyro.
    """
    # https://stackoverflow.com/questions/32808686/storing-a-random-state/32809283

    file_dict = {}

    # Random state information
    if USE_PYRO:
        pyro_random_state = pyro.util.get_rng_state()  # this is shorthand for the following
        file_dict.update({filebase + "_random.pyro": pyro_random_state})
    else:
        python_random_state = random.getstate()
        numpy_random_state = np.random.get_state()
        torch_random_state = torch.get_rng_state()
        file_dict.update(
            {
                filebase + "_random.python": python_random_state,
                filebase + "_random.numpy": numpy_random_state,
                filebase + "_random.torch": torch_random_state,
            }
        )
    if USE_CUDA:
        cuda_random_state = torch.cuda.get_rng_state_all()
        file_dict.update({filebase + "_random.cuda": cuda_random_state})

    # Save it
    for file, state in file_dict.items():
        with open(file, "wb") as f:
            pickle.dump(state, f)

    return list(file_dict.keys())


def load_random_state(filebase: str):
    """Load random states from files and update generators with them."""

    if USE_PYRO:
        with open(filebase + "_random.pyro", "rb") as f:
            pyro_random_state = pickle.load(f)
        pyro.util.set_rng_state(pyro_random_state)

    else:
        with open(filebase + "_random.python", "rb") as f:
            python_random_state = pickle.load(f)
        random.setstate(python_random_state)

        with open(filebase + "_random.numpy", "rb") as f:
            numpy_random_state = pickle.load(f)
        np.random.set_state(numpy_random_state)

        with open(filebase + "_random.torch", "rb") as f:
            torch_random_state = pickle.load(f)
        torch.set_rng_state(torch_random_state)

    if USE_CUDA:
        with open(filebase + "_random.cuda", "rb") as f:
            cuda_random_state = pickle.load(f)
        torch.cuda.set_rng_state_all(cuda_random_state)


def _serialize_checkpoint_to_buffers(
    filebase: str,
    model_obj: "RemoveBackgroundPyroModel",
    scheduler: pyro.optim.PyroOptim,
    args: argparse.Namespace,
    train_loader: Optional[DataLoader] = None,
    test_loader: Optional[DataLoader] = None,
) -> Dict[str, bytes]:
    """Serialize all checkpoint state to in-memory byte buffers on the calling thread.

    This is the CPU-bound capture phase of async checkpointing. After this
    returns, the caller is free to resume training; the background thread can
    write the bytes to disk without touching any shared mutable state.

    Returns:
        Dict mapping bare filename to bytes content, ready to be written to a
        tarball by :func:`_write_checkpoint_buffers`.
    """
    buffers: Dict[str, bytes] = {}
    basename = os.path.basename(filebase)

    # Random state — save_random_state requires a file path.
    with tempfile.TemporaryDirectory() as rng_tmp:
        tmp_base = os.path.join(rng_tmp, basename)
        rng_files = save_random_state(filebase=tmp_base)
        for path in rng_files:
            with open(path, "rb") as f:
                buffers[os.path.basename(path)] = f.read()

    # Phase 2: model state_dict + JSON metadata.
    buf = io.BytesIO()
    torch.save(model_obj.state_dict(), buf)
    buffers[basename + "_model.state_dict.torch"] = buf.getvalue()
    meta = _build_model_meta_dict(model_obj)
    if meta is not None:
        buffers[basename + "_model.meta.json"] = json.dumps(meta).encode()

    # Phase 3: optimizer pyro-format only (requires a real file path).
    ntf_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(suffix="_optim.pyro", delete=False) as ntf:
            ntf_path = ntf.name
        scheduler.save(ntf_path)
        with open(ntf_path, "rb") as f:
            buffers[basename + "_optim.pyro"] = f.read()
    finally:
        if ntf_path is not None:
            try:
                os.unlink(ntf_path)
            except OSError:
                pass

    # Param store — only scalar (non-module) params.  Encoder / decoder
    # nn.Parameter tensors are saved via model.state_dict.torch and must NOT
    # also be saved here to avoid the two-tensor-object divergence on restore.
    buf = io.BytesIO()
    torch.save(_get_scalar_param_store_state(model_obj), buf, pickle_module=dill)
    buffers[basename + "_params.pyro"] = buf.getvalue()

    # Phase 1: DataLoaders as compact npz buffers.
    if train_loader is not None:
        buf = io.BytesIO()
        np.savez_compressed(buf, **train_loader.get_state())
        buffers[basename + "_train.loaderstate.npz"] = buf.getvalue()
    if test_loader is not None:
        buf = io.BytesIO()
        np.savez_compressed(buf, **test_loader.get_state())
        buffers[basename + "_test.loaderstate.npz"] = buf.getvalue()

    # Args.
    buf = io.BytesIO()
    np.save(buf, args)
    buffers[basename + "_args.npy"] = buf.getvalue()

    return buffers


def _write_checkpoint_buffers(buffers: Dict[str, bytes], tarball_name: str) -> bool:
    """Write buffered checkpoint bytes to disk and create a tarball.

    Intended to run in a background thread started by
    :func:`save_checkpoint_async`.
    """
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_list = []
            for name, data in buffers.items():
                fpath = os.path.join(tmp_dir, name)
                with open(fpath, "wb") as f:
                    f.write(data)
                file_list.append(fpath)
            make_tarball(files=file_list, tarball_name=tarball_name)
        logger.info(f"Checkpoint written asynchronously to {os.path.abspath(tarball_name)}")
        return True
    except Exception:
        logger.warning("Async checkpoint write failed")
        logger.warning(traceback.format_exc())
        return False


def save_checkpoint(
    filebase: str,
    model_obj: "RemoveBackgroundPyroModel",
    scheduler: pyro.optim.PyroOptim,
    args: argparse.Namespace,
    train_loader: Optional[DataLoader] = None,
    test_loader: Optional[DataLoader] = None,
    tarball_name: str = consts.CHECKPOINT_FILE_NAME,
) -> bool:
    """Save trained model and optimizer state in a checkpoint file.
    Any hyperparameters or metadata should be part of the model object."""

    logger.info("Saving a checkpoint...")

    try:
        # Work in a temporary directory.
        with tempfile.TemporaryDirectory() as tmp_dir:
            basename = os.path.basename(filebase)
            filebase = os.path.join(tmp_dir, basename)

            file_list = save_random_state(filebase=filebase)

            # Phase 2: model state_dict + JSON metadata (no dill pickle).
            torch.save(model_obj.state_dict(), filebase + "_model.state_dict.torch")
            _save_model_meta(model_obj, filebase + "_model.meta.json")
            file_list += [
                filebase + "_model.state_dict.torch",
            ]
            if os.path.exists(filebase + "_model.meta.json"):
                file_list.append(filebase + "_model.meta.json")

            # Phase 3: optimizer pyro-format only (drop full torch pickle).
            # Save only scalar (non-module) params to avoid the two-tensor-object
            # divergence that arises when encoder/decoder params are serialised
            # separately from model.state_dict.torch.
            scheduler.save(filebase + "_optim.pyro")
            scalar_ps_state = _get_scalar_param_store_state(model_obj)
            with open(filebase + "_params.pyro", "wb") as _f:
                torch.save(scalar_ps_state, _f, pickle_module=dill)
            file_list += [
                filebase + "_optim.pyro",
                filebase + "_params.pyro",
            ]

            # Phase 1: DataLoaders as compact index-based npz (no sparse matrices).
            if train_loader is not None:
                np.savez_compressed(filebase + "_train.loaderstate.npz", **train_loader.get_state())
                file_list.append(filebase + "_train.loaderstate.npz")
            if test_loader is not None:
                np.savez_compressed(filebase + "_test.loaderstate.npz", **test_loader.get_state())
                file_list.append(filebase + "_test.loaderstate.npz")

            np.save(filebase + "_args.npy", args)
            file_list.append(filebase + "_args.npy")

            make_tarball(files=file_list, tarball_name=tarball_name)

        logger.info(f"Saved checkpoint as {os.path.abspath(tarball_name)}")
        return True

    except KeyboardInterrupt:
        logger.warning("Keyboard interrupt: will not save checkpoint")
        return False

    except Exception:
        logger.warning("Could not save checkpoint")
        logger.warning(traceback.format_exc())
        return False


def save_checkpoint_async(
    filebase: str,
    model_obj: "RemoveBackgroundPyroModel",
    scheduler: pyro.optim.PyroOptim,
    args: argparse.Namespace,
    train_loader: Optional[DataLoader] = None,
    test_loader: Optional[DataLoader] = None,
    tarball_name: str = consts.CHECKPOINT_FILE_NAME,
) -> Optional[concurrent.futures.Future]:
    """Serialize checkpoint state on the calling thread, then write to disk in
    a background thread so training can resume immediately.

    The serialization phase (the slow part on CPU — model pickle + gzip) still
    blocks the caller briefly, but the I/O-heavy tarball write happens off the
    training thread.  Call :func:`flush_pending_checkpoint` before reading the
    output file to guarantee the write has completed.

    If serialization raises, this falls back transparently to
    :func:`save_checkpoint` so callers never need to check for None in the
    normal path.

    Returns:
        The :class:`concurrent.futures.Future` for the background write, or
        ``None`` if serialization failed (sync fallback was used instead).
    """
    global _pending_ckpt

    # Wait for any previous background write before capturing a new snapshot.
    if _pending_ckpt is not None:
        try:
            _pending_ckpt.result()
        except Exception:
            logger.warning("Previous async checkpoint write raised an exception:")
            logger.warning(traceback.format_exc())
        _pending_ckpt = None

    logger.info("Serializing checkpoint state...")
    try:
        buffers = _serialize_checkpoint_to_buffers(
            filebase=filebase,
            model_obj=model_obj,
            scheduler=scheduler,
            args=args,
            train_loader=train_loader,
            test_loader=test_loader,
        )
    except KeyboardInterrupt:
        logger.warning("Keyboard interrupt during checkpoint serialization: not saving")
        return None
    except Exception:
        logger.warning("Could not serialize checkpoint state; falling back to synchronous save")
        logger.warning(traceback.format_exc())
        save_checkpoint(
            filebase=filebase,
            model_obj=model_obj,
            scheduler=scheduler,
            args=args,
            train_loader=train_loader,
            test_loader=test_loader,
            tarball_name=tarball_name,
        )
        return None

    _pending_ckpt = _ckpt_executor.submit(_write_checkpoint_buffers, buffers, tarball_name)
    logger.info(f"Writing checkpoint to {os.path.abspath(tarball_name)} in background")
    return _pending_ckpt


def flush_pending_checkpoint() -> bool:
    """Block until any in-flight async checkpoint write completes.

    Call this after the training loop (and before reading the checkpoint file)
    to guarantee the last :func:`save_checkpoint_async` has landed on disk.

    Returns:
        ``True`` if the pending checkpoint succeeded (or nothing was pending),
        ``False`` if the background write raised an exception.
    """
    global _pending_ckpt
    if _pending_ckpt is None:
        return True
    try:
        result = _pending_ckpt.result()
        return bool(result)
    except Exception:
        logger.warning("Async checkpoint write failed:")
        logger.warning(traceback.format_exc())
        return False
    finally:
        _pending_ckpt = None


def load_checkpoint(
    filebase: Optional[str],
    tarball_name: str = consts.CHECKPOINT_FILE_NAME,
    force_device: Optional[str] = None,
    force_use_checkpoint: bool = False,
) -> Dict:
    """Load checkpoint and return state for model reconstruction.

    Returns a dict with keys:
        ``model_state_dict``, ``model_meta``, ``optim_state_bytes``,
        ``train_loader_state``, ``test_loader_state``, ``args``,
        ``loaded`` (always True on success).
    Raises FileNotFoundError or ValueError on failure.
    """
    out = load_from_checkpoint(
        filebase=filebase,
        tarball_name=tarball_name,
        to_load=["model", "optim", "param_store", "dataloader", "args", "random_state"],
        force_device=force_device,
        force_use_checkpoint=force_use_checkpoint,
    )
    out["loaded"] = True
    logger.info(f"Loaded partially-trained checkpoint from {tarball_name}")
    return out


def load_from_checkpoint(
    filebase: Optional[str],
    tarball_name: str = consts.CHECKPOINT_FILE_NAME,
    to_load: List[str] = ["model"],
    force_device: Optional[str] = None,
    force_use_checkpoint: bool = False,
) -> Dict:
    """Load specific files from a checkpoint tarball."""

    load_kwargs: dict[str, Any] = {}
    map_location: torch.device | None = None
    if force_device is not None:
        map_location = torch.device(force_device)
        load_kwargs["map_location"] = map_location

    # Work in a temporary directory.
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Unpack the checkpoint tarball.
        logger.info(f'Attempting to unpack tarball "{tarball_name}" to {tmp_dir}')
        success = unpack_tarball(tarball_name=tarball_name, directory=tmp_dir)
        if success:
            unpacked_files = "\n".join(glob.glob(os.path.join(tmp_dir, "*")))
            logger.info(f"Successfully unpacked tarball to {tmp_dir}\n{unpacked_files}")
        else:
            # no tarball loaded, so do not continue trying to load files
            raise FileNotFoundError

        # If posterior is present, do not require run hash to match: will pick up
        # after training and run estimation from existing posterior.
        # This smoothly allows re-runs (including for problematic v0.3.1)
        logger.debug(f"force_use_checkpoint: {force_use_checkpoint}")
        if force_use_checkpoint or (filebase is None):
            filebase = glob.glob(os.path.join(tmp_dir, "*_model.state_dict.torch"))[0].replace(
                "_model.state_dict.torch", ""
            )
            logger.debug(f"Accepting any file hash, so loading {filebase}*")

        else:
            # See if files have a hash matching input filebase.
            basename = os.path.basename(filebase)
            filebase = os.path.join(tmp_dir, basename)
            logger.debug(f"Looking for files with base name matching {filebase}*")
            if not os.path.exists(filebase + "_model.state_dict.torch"):
                logger.info("Workflow hash does not match that of checkpoint.")
                raise ValueError("Workflow hash does not match that of checkpoint.")

        out = {}

        # Load the saved model state_dict + metadata JSON.
        if "model" in to_load:
            state_dict = torch.load(
                filebase + "_model.state_dict.torch", map_location=map_location, weights_only=True
            )
            out["model_state_dict"] = state_dict
            meta_path = filebase + "_model.meta.json"
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    out["model_meta"] = json.load(f)
            logger.debug("Model state_dict loaded from " + filebase + "_model.state_dict.torch")

        # Load the saved optimizer state bytes (pyro format only).
        if "optim" in to_load:
            with open(filebase + "_optim.pyro", "rb") as f:
                out["optim_state_bytes"] = f.read()
            logger.debug("Optimizer state loaded from " + filebase + "_optim.pyro")

        # Load the pyro param store.
        if "param_store" in to_load:
            load_param_store(filebase + "_params.pyro", force_device)
            logger.debug("Pyro param store loaded from " + filebase + "_params.pyro")

        # Load dataloader states (compact npz format).
        if "dataloader" in to_load:
            train_npz = filebase + "_train.loaderstate.npz"
            test_npz = filebase + "_test.loaderstate.npz"
            if os.path.exists(train_npz):
                out["train_loader_state"] = dict(np.load(train_npz, allow_pickle=False))
                logger.debug("Train loader state loaded from " + train_npz)
            if os.path.exists(test_npz):
                out["test_loader_state"] = dict(np.load(test_npz, allow_pickle=False))
                logger.debug("Test loader state loaded from " + test_npz)

        # Load args, which can be modified in the case of auto-learning-rate updates.
        if "args" in to_load:
            args = np.load(filebase + "_args.npy", allow_pickle=True).item()
            out.update({"args": args})

        # Update states of random number generators across the board.
        if "random_state" in to_load:
            load_random_state(filebase=filebase)
            logger.debug("Loaded random state globally for python, numpy, pytorch, and cuda")

        # Copy the posterior file outside the temp dir so it can be loaded later.
        if "posterior" in to_load:
            posterior_file = os.path.join(os.path.dirname(filebase), "posterior.parquet")
            if os.path.exists(posterior_file):
                shutil.copyfile(posterior_file, "posterior.parquet")
                out.update({"posterior_file": "posterior.parquet"})
                logger.debug(f"Copied posterior file from {posterior_file} to posterior.parquet")
                # Also copy latents CSV sidecar if present.
                latents_src = os.path.join(os.path.dirname(filebase), "posterior_latents.csv.gz")
                if os.path.exists(latents_src):
                    shutil.copyfile(latents_src, "posterior_latents.csv.gz")
            else:
                logger.debug(
                    "Trying to load posterior in load_from_checkpoint(), but posterior "
                    "is not present in checkpoint tarball."
                )

    return out


def attempt_load_checkpoint(
    filebase: Optional[str],
    tarball_name: str = consts.CHECKPOINT_FILE_NAME,
    force_device: Optional[str] = None,
    force_use_checkpoint: bool = False,
) -> Dict[str, Union["RemoveBackgroundPyroModel", pyro.optim.PyroOptim, DataLoader, bool]]:
    """Load checkpoint and prepare a RemoveBackgroundPyroModel and optimizer,
    or return the inputs if loading fails."""

    try:
        logger.debug("Attempting to load checkpoint from " + tarball_name)
        return load_checkpoint(
            filebase=filebase,
            tarball_name=tarball_name,
            force_device=force_device,
            force_use_checkpoint=force_use_checkpoint,
        )

    except FileNotFoundError:
        logger.debug("No tarball found")
        return {"loaded": False}

    except ValueError:
        logger.debug("Unpacked tarball files have a different workflow hash: will not load")
        return {"loaded": False}


def make_tarball(files: List[str], tarball_name: str) -> bool:
    """Tar and gzip a list of files as a checkpoint tarball.
    NOTE: used by automated checkpoint handling in Cromwell
    NOTE2: .tmp file is used so that incomplete checkpoint files do not exist
        even temporarily
    """
    with tarfile.open(tarball_name + ".tmp", "w:gz", compresslevel=1) as tar:
        for file in files:
            # without arcname, unpacking results in unpredictable file locations!
            tar.add(file, arcname=os.path.basename(file))
    os.replace(tarball_name + ".tmp", tarball_name)
    return True


def unpack_tarball(tarball_name: str, directory: str) -> bool:
    """Untar a checkpoint tarball and put the files in directory."""
    if not os.path.exists(tarball_name):
        logger.info("No saved checkpoint.")
        return False

    try:
        with tarfile.open(tarball_name, "r:gz") as tar:
            tar.extractall(path=directory)
        return True

    except Exception:
        logger.warning("Failed to unpack existing tarball.")
        return False


def create_workflow_hashcode(
    module_path: str,
    args: argparse.Namespace,
    args_to_remove: List[str] = ["epochs", "fpr"],
    name: str = "md5",
    verbose: bool = False,
) -> str:
    """Create a hash blob from cellbender python code plus input arguments."""

    hasher = hashlib.new(name=name)

    files_safe_to_ignore = [
        "infer.py",
        "simulate.py",
        "report.py",
        "downstream.py",
        "monitor.py",
        "fpr.py",
        "posterior.py",
        "estimation.py",
        "sparse_utils.py",
    ]

    if not os.path.exists(module_path):
        return ""

    try:
        # files
        for root, _, files in os.walk(module_path):
            for file in files:
                if not file.endswith(".py"):
                    continue
                if file in files_safe_to_ignore:
                    continue
                if "test" in file:
                    continue
                if verbose:
                    print(file)
                with open(os.path.join(root, file), "rb") as f:
                    buf = b"\n".join(f.readlines())
                hasher.update(buf)  # encode python files

        # inputs
        args_dict = vars(args).copy()
        for arg in args_to_remove:
            args_dict.pop(arg, None)
        hasher.update(str(args_dict).encode("utf-8"))  # encode parsed input args

        # input file
        # TODO
        # this is probably not necessary for real data... why would two different
        # files have the same name?
        # but it's useful for development where simulated datasets change

    except Exception:
        return ""

    return hasher.hexdigest()


def save_param_store(filename: str) -> None:
    """Save parameters to file."""

    # Modified from pyro to allow alternate pickle_module
    with open(filename, "wb") as output_file:
        torch.save(pyro.get_param_store().get_state(), output_file, pickle_module=dill)


def load_param_store(filename: str, force_device: Optional[str] = None) -> None:
    """Load parameters to file."""

    # Modified from pyro to allow alternate pickle_module
    with open(filename, "rb") as input_file:
        state = torch.load(input_file, force_device, pickle_module=dill)
    pyro.get_param_store().set_state(state)
