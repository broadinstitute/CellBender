"""Handle the saving and loading of models from checkpoint files."""

from cellbender.remove_background import consts
from cellbender.remove_background.data.dataprep import DataLoader

import torch
import numpy as np
from typing import Tuple, List, Optional, Union, Dict
import logging
import argparse
import hashlib
import os
import tarfile
import glob
import random
import pickle
import tempfile
import shutil
import traceback


logger = logging.getLogger('cellbender')

USE_PYRO = True
try:
    import pyro
except ImportError:
    USE_PYRO = False
USE_CUDA = torch.cuda.is_available()


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
        file_dict.update({filebase + '_random.pyro': pyro_random_state})
    else:
        python_random_state = random.getstate()
        numpy_random_state = np.random.get_state()
        torch_random_state = torch.get_rng_state()
        file_dict.update({filebase + '_random.python': python_random_state,
                          filebase + '_random.numpy': numpy_random_state,
                          filebase + '_random.torch': torch_random_state})
    if USE_CUDA:
        cuda_random_state = torch.cuda.get_rng_state_all()
        file_dict.update({filebase + '_random.cuda': cuda_random_state})

    # Save it
    for file, state in file_dict.items():
        with open(file, 'wb') as f:
            pickle.dump(state, f)

    return list(file_dict.keys())


def load_random_state(filebase: str):
    """Load random states from files and update generators with them."""

    if USE_PYRO:
        with open(filebase + '_random.pyro', 'rb') as f:
            pyro_random_state = pickle.load(f)
        pyro.util.set_rng_state(pyro_random_state)

    else:
        with open(filebase + '_random.python', 'rb') as f:
            python_random_state = pickle.load(f)
        random.setstate(python_random_state)

        with open(filebase + '_random.numpy', 'rb') as f:
            numpy_random_state = pickle.load(f)
        np.random.set_state(numpy_random_state)

        with open(filebase + '_random.torch', 'rb') as f:
            torch_random_state = pickle.load(f)
        torch.set_rng_state(torch_random_state)

    if USE_CUDA:
        with open(filebase + '_random.cuda', 'rb') as f:
            cuda_random_state = pickle.load(f)
        torch.cuda.set_rng_state_all(cuda_random_state)


def save_checkpoint(filebase: str,
                    model_obj: 'RemoveBackgroundPyroModel',
                    scheduler: pyro.optim.PyroOptim,
                    args: argparse.Namespace,
                    train_loader: Optional[DataLoader] = None,
                    test_loader: Optional[DataLoader] = None,
                    tarball_name: str = consts.CHECKPOINT_FILE_NAME) -> bool:
    """Save trained model and optimizer state in a checkpoint file.
    Any hyperparameters or metadata should be part of the model object."""

    logger.info(f'Saving a checkpoint...')

    try:

        # Work in a temporary directory.
        with tempfile.TemporaryDirectory() as tmp_dir:

            basename = os.path.basename(filebase)
            filebase = os.path.join(tmp_dir, basename)

            file_list = save_random_state(filebase=filebase)

            torch.save(model_obj.state_dict(), filebase + '_model.torch')
            scheduler.save(filebase + '_optim.torch')
            scheduler.save(filebase + '_optim.pyro')  # use PyroOptim method
            pyro.get_param_store().save(filebase + '_params.pyro')
            file_list = file_list + [filebase + '_model.torch',
                                     filebase + '_optim.torch',
                                     filebase + '_optim.pyro',
                                     filebase + '_params.pyro']

            if train_loader is not None:
                # train_loader_file = save_dataloader_state(filebase=filebase,
                #                                           data_loader_state=train_loader.get_state(),
                #                                           name='train')
                torch.save(train_loader, filebase + '_train.loaderstate')
                file_list.append(filebase + '_train.loaderstate')
            if test_loader is not None:
                # test_loader_file = save_dataloader_state(filebase=filebase,
                #                                          data_loader_state=test_loader.get_state(),
                #                                          name='test')
                torch.save(test_loader, filebase + '_test.loaderstate')
                file_list.append(filebase + '_test.loaderstate')

            np.save(filebase + '_args.npy', args)
            file_list.append(filebase + '_args.npy')

            make_tarball(files=file_list, tarball_name=tarball_name)

        logger.info(f'Saved checkpoint as {os.path.abspath(tarball_name)}')
        return True

    except KeyboardInterrupt:
        logger.warning('Keyboard interrupt: will not save checkpoint')
        return False

    except Exception:
        logger.warning('Could not save checkpoint')
        logger.warning(traceback.format_exc())
        return False


def load_checkpoint(filebase: Optional[str],
                    tarball_name: str = consts.CHECKPOINT_FILE_NAME,
                    force_device: Optional[str] = None,
                    force_use_checkpoint: bool = False)\
        -> Dict[str, Union['RemoveBackgroundPyroModel', pyro.optim.PyroOptim, DataLoader, bool]]:
    """Load checkpoint and prepare a RemoveBackgroundPyroModel and optimizer."""

    out = load_from_checkpoint(
        filebase=filebase,
        tarball_name=tarball_name,
        to_load=['model', 'optim', 'param_store', 'dataloader', 'args', 'random_state'],
        force_device=force_device,
        force_use_checkpoint=force_use_checkpoint,
    )
    out.update({'loaded': True})
    logger.info(f'Loaded partially-trained checkpoint from {tarball_name}')
    return out


def load_from_checkpoint(filebase: Optional[str],
                         tarball_name: str = consts.CHECKPOINT_FILE_NAME,
                         to_load: List[str] = ['model'],
                         force_device: Optional[str] = None,
                         force_use_checkpoint: bool = False) -> Dict:
    """Load specific files from a checkpoint tarball."""

    load_kwargs = {}
    if force_device is not None:
        load_kwargs.update({'map_location': torch.device(force_device)})

    # Work in a temporary directory.
    with tempfile.TemporaryDirectory() as tmp_dir:

        # Unpack the checkpoint tarball.
        logger.info(f'Attempting to unpack tarball "{tarball_name}" to {tmp_dir}')
        success = unpack_tarball(tarball_name=tarball_name, directory=tmp_dir)
        if success:
            unpacked_files = '\n'.join(glob.glob(os.path.join(tmp_dir, "*")))
            logger.info(f'Successfully unpacked tarball to {tmp_dir}\n'
                        f'{unpacked_files}')
        else:
            # no tarball loaded, so do not continue trying to load files
            raise FileNotFoundError
        
        # If posterior is present, do not require run hash to match: will pick up 
        # after training and run estimation from existing posterior.
        # This smoothly allows re-runs (including for problematic v0.3.1)
        logger.debug(f'force_use_checkpoint: {force_use_checkpoint}')
        if force_use_checkpoint or (filebase is None):
            filebase = (glob.glob(os.path.join(tmp_dir, '*_model.torch'))[0]
                        .replace('_model.torch', ''))
            logger.debug(f'Accepting any file hash, so loading {filebase}*')
        
        else:
            # See if files have a hash matching input filebase.
            basename = os.path.basename(filebase)
            filebase = os.path.join(tmp_dir, basename)
            logger.debug(f'Looking for files with base name matching {filebase}*')
            if not os.path.exists(filebase + '_model.torch'):
                logger.info('Workflow hash does not match that of checkpoint.')
                raise ValueError('Workflow hash does not match that of checkpoint.')            

        out = {}

        # Load the saved model.
        if 'model' in to_load:
            model_obj = torch.load(filebase + '_model.torch', **load_kwargs)
            logger.debug('Model loaded from ' + filebase + '_model.torch')
            out.update({'model': model_obj})

        # Load the saved optimizer.
        if 'optim' in to_load:
            scheduler = torch.load(filebase + '_optim.torch', **load_kwargs)
            scheduler.load(filebase + '_optim.pyro', **load_kwargs)  # use PyroOptim method
            logger.debug('Optimizer loaded from ' + filebase + '_optim.*')
            out.update({'optim': scheduler})

        # Load the pyro param store.
        if 'param_store' in to_load:
            pyro.get_param_store().load(filebase + '_params.pyro', map_location=force_device)
            logger.debug('Pyro param store loaded from ' + filebase + '_params.pyro')

        # Load dataloader states.
        if 'dataloader' in to_load:
            # load_dataloader_state(data_loader=train_loader, file=filebase + '_train.loaderstate')
            # load_dataloader_state(data_loader=test_loader, file=filebase + '_test.loaderstate')
            train_loader = None
            test_loader = None
            if os.path.exists(filebase + '_train.loaderstate'):
                train_loader = torch.load(filebase + '_train.loaderstate', **load_kwargs)
                logger.debug('Train loader loaded from ' + filebase + '_train.loaderstate')
                out.update({'train_loader': train_loader})
            if os.path.exists(filebase + '_test.loaderstate'):
                test_loader = torch.load(filebase + '_test.loaderstate', **load_kwargs)
                logger.debug('Test loader loaded from ' + filebase + '_test.loaderstate')
                out.update({'test_loader': test_loader})

        # Load args, which can be modified in the case of auto-learning-rate updates.
        if 'args' in to_load:
            args = np.load(filebase + '_args.npy', allow_pickle=True).item()
            out.update({'args': args})

        # Update states of random number generators across the board.
        if 'random_state' in to_load:
            load_random_state(filebase=filebase)
            logger.debug('Loaded random state globally for python, numpy, pytorch, and cuda')

        # Copy the posterior file outside the temp dir so it can be loaded later.
        if 'posterior' in to_load:
            posterior_file = os.path.join(os.path.dirname(filebase), 'posterior.h5')
            if os.path.exists(posterior_file):
                shutil.copyfile(posterior_file, 'posterior.h5')
                out.update({'posterior_file': 'posterior.h5'})
                logger.debug(f'Copied posterior file from {posterior_file} to posterior.h5')
            else:
                logger.debug('Trying to load posterior in load_from_checkpoint(), but posterior '
                             'is not present in checkpoint tarball.')

    return out


def attempt_load_checkpoint(filebase: Optional[str],
                            tarball_name: str = consts.CHECKPOINT_FILE_NAME,
                            force_device: Optional[str] = None,
                            force_use_checkpoint: bool = False)\
        -> Dict[str, Union['RemoveBackgroundPyroModel', pyro.optim.PyroOptim, DataLoader, bool]]:
    """Load checkpoint and prepare a RemoveBackgroundPyroModel and optimizer,
    or return the inputs if loading fails."""

    try:
        logger.debug('Attempting to load checkpoint from ' + tarball_name)
        return load_checkpoint(filebase=filebase,
                               tarball_name=tarball_name,
                               force_device=force_device,
                               force_use_checkpoint=force_use_checkpoint)

    except FileNotFoundError:
        logger.debug('No tarball found')
        return {'loaded': False}

    except ValueError:
        logger.debug('Unpacked tarball files have a different workflow hash: will not load')
        return {'loaded': False}


def make_tarball(files: List[str], tarball_name: str) -> bool:
    """Tar and gzip a list of files as a checkpoint tarball.
    NOTE: used by automated checkpoint handling in Cromwell
    NOTE2: .tmp file is used so that incomplete checkpoint files do not exist
        even temporarily
    """
    with tarfile.open(tarball_name + '.tmp', 'w:gz', compresslevel=1) as tar:
        for file in files:
            # without arcname, unpacking results in unpredictable file locations!
            tar.add(file, arcname=os.path.basename(file))
    os.replace(tarball_name + '.tmp', tarball_name)
    return True


def unpack_tarball(tarball_name: str, directory: str) -> bool:
    """Untar a checkpoint tarball and put the files in directory."""
    if not os.path.exists(tarball_name):
        logger.info("No saved checkpoint.")
        return False

    try:
        with tarfile.open(tarball_name, 'r:gz') as tar:
            tar.extractall(path=directory)
        return True

    except Exception:
        logger.warning("Failed to unpack existing tarball.")
        return False


def create_workflow_hashcode(module_path: str,
                             args: argparse.Namespace,
                             args_to_remove: List[str] = ['epochs', 'fpr'],
                             name: str = 'md5',
                             verbose: bool = False) -> str:
    """Create a hash blob from cellbender python code plus input arguments."""

    hasher = hashlib.new(name=name)

    files_safe_to_ignore = ['infer.py', 'simulate.py', 'report.py',
                            'downstream.py', 'monitor.py', 'fpr.py',
                            'posterior.py', 'estimation.py', 'sparse_utils.py']

    if not os.path.exists(module_path):
        return ''

    try:

        # files
        for root, _, files in os.walk(module_path):
            for file in files:
                if not file.endswith('.py'):
                    continue
                if file in files_safe_to_ignore:
                    continue
                if 'test' in file:
                    continue
                if verbose:
                    print(file)
                with open(os.path.join(root, file), 'rb') as f:
                    buf = b'\n'.join(f.readlines())
                hasher.update(buf)  # encode python files

        # inputs
        args_dict = vars(args).copy()
        for arg in args_to_remove:
            args_dict.pop(arg, None)
        hasher.update(str(args_dict).encode('utf-8'))  # encode parsed input args

        # input file
        # TODO
        # this is probably not necessary for real data... why would two different
        # files have the same name?
        # but it's useful for development where simulated datasets change

    except Exception:
        return ''

    return hasher.hexdigest()
