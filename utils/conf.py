
"""
This module contains utility functions for configuration settings.
"""

# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import gc
import logging
import random
from functools import partial

from typing import List, Optional, Union
import numpy as np
import torch
from torch.utils.data import DataLoader


def warn_once(*msg):
    """
    Prints a warning message only once.

    Args:
        msg: the message to be printed
    """
    msg = ' '.join([str(m) for m in msg])
    if not hasattr(warn_once, 'warned'):
        warn_once.warned = set() # type: ignore[unresolved-attribute]
    if msg not in warn_once.warned: # type: ignore[unresolved-attribute]
        warn_once.warned.add(msg) # type: ignore[unresolved-attribute]
        logging.warning(msg)


def _get_gpu_memory_pynvml_all_processes(device_id: int = 0) -> int:
    """
    Use pynvml to get the memory allocated on the GPU.
    Returns the memory allocated on the GPU in Bytes.
    """
    if not hasattr(_get_gpu_memory_pynvml_all_processes, f'handle_{device_id}'):
        torch.cuda.pynvml.nvmlInit() # type: ignore[possibly-unbound-attribute]
        handle = torch.cuda.pynvml.nvmlDeviceGetHandleByIndex(device_id) # type: ignore[possibly-unbound-attribute]
        setattr(_get_gpu_memory_pynvml_all_processes, f'handle_{device_id}', handle)

    handle = getattr(_get_gpu_memory_pynvml_all_processes, f'handle_{device_id}')

    procs = torch.cuda.pynvml.nvmlDeviceGetComputeRunningProcesses(handle) # type: ignore[possibly-unbound-attribute]
    return sum([proc.usedGpuMemory for proc in procs])


from typing import List, Tuple, Union

def get_alloc_memory_all_devices(avail_devices=None, return_all=False) -> Union[List[int], Tuple[List[int], List[int], List[int]]]:
    """
    Returns the memory allocated on all the available devices.
    By default, tries to return the memory read from pynvml, if available.
    Else, it returns the memory `reserved` by torch.

    If `return_all` is set to True, it returns a tuple with the memory reserved, allocated and from pynvml.

    If `avail_devices` is set, it returns the memory only for the specified devices.

    Values are in Bytes.
    """
    gpu_memory_reserved = []
    gpu_memory_allocated = []
    gpu_memory_nvidiasmi = []
    for i in range(torch.cuda.device_count()):
        if avail_devices is not None and i not in avail_devices:
            continue

        _ = torch.tensor([1]).to(i)  # allocate memory to get more accurate reading from torch
        gpu_memory_reserved.append(torch.cuda.max_memory_reserved(i))
        gpu_memory_allocated.append(torch.cuda.max_memory_allocated(i))

        try:
            gpu_memory_nvidiasmi.append(_get_gpu_memory_pynvml_all_processes(i))
        except BaseException as e:
            warn_once('Could not get memory from pynvml. Maybe try `pip install --force-reinstall gpustat`.', str(e))
            gpu_memory_nvidiasmi.append(-1)

        del _
        gc.collect()
        torch.cuda.empty_cache()

    if return_all:
        return gpu_memory_reserved, gpu_memory_allocated, gpu_memory_nvidiasmi
    else:
        if any([g > 0 for g in gpu_memory_nvidiasmi]):
            return gpu_memory_nvidiasmi
        return gpu_memory_allocated


def get_device(avail_devices: Optional[str] = None) -> torch.device:
    """
    Returns the least used GPU device if available else MPS or CPU.
    """
    def _get_device(avail_devices: List[int] = []) -> torch.device:
        # get least used gpu by used memory
        if torch.cuda.is_available() and torch.cuda.device_count() > 0 and len(avail_devices) > 0:
            gpu_memory = get_alloc_memory_all_devices(avail_devices=avail_devices)
            device = torch.device(f'cuda:{avail_devices[np.argmin(gpu_memory)]}')
            return device
        try:
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                logging.warning("MSP support is still experimental. Use at your own risk!")
                return torch.device("mps")
        except BaseException:
            logging.error("Something went wrong with MPS. Using CPU.")

        logging.warning("No GPU available. Using CPU.")
        return torch.device("cpu")

    # Permanently store the chosen device
    if not hasattr(get_device, 'device'):
        if avail_devices is not None:
            devices_list = [int(d.strip(':')[-1]) for d in avail_devices.split(',')]
        else:
            devices_list = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []

        get_device.device = _get_device(avail_devices=devices_list) # type: ignore[unresolved-attribute]
        logging.info(f'Using device {get_device.device}')

    return get_device.device # type: ignore[unresolved-attribute]


def base_path(override=None) -> str:
    """
    Returns the base bath where to log accuracies and tensorboard data.

    Args:
        override: the path to override the default one. Once set, it is stored and used for all the next calls.

    Returns:
        the base path (default: `./data/`)
    """
    if override is not None:
        if not os.path.exists(override):
            os.makedirs(override)
        if not override.endswith('/'):
            override += '/'
        setattr(base_path, 'path', override)

    if not hasattr(base_path, 'path'):
        setattr(base_path, 'path', './data/')
    return getattr(base_path, 'path')

def get_checkpoint_path(override=None) -> str:
    """
    Returns the path where to save the checkpoints.

    Args:
        override: the path to override the default one. Once set, it is stored and used for all the next calls.

    Returns:
        the checkpoint path (default: `./checkpoints/`)
    """
    if override is not None:
        if not os.path.exists(override):
            os.makedirs(override)
        if not override.endswith('/'):
            override += '/'
        setattr(get_checkpoint_path, 'path', override)

    if not hasattr(get_checkpoint_path, 'path'):
        setattr(get_checkpoint_path, 'path', './checkpoints/')
    return getattr(get_checkpoint_path, 'path')

def set_random_seed(seed: int) -> None:
    """
    Sets the seeds at a certain value.

    Args:
        seed: the value to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except BaseException:
        logging.error('Could not set cuda seed.')


def worker_init_fn(worker_id, num_workers, seed, rank=1):
    """
    Sets the seeds for a worker of a dataloader.
    The seed of each worker is set to: `num_worker * rank + worker_id + seed`
    """
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_seeded_dataloader(args, dataset, non_verbose=False, **dataloader_args) -> DataLoader:
    """
    Creates a dataloader object from a dataset, setting the seeds for the workers (if `--seed` is set).

    Args:
        args: the arguments of the program
        dataset: the dataset to be loaded
        verbose: whether to print the number of workers
        dataloader_args: external arguments of the dataloader

    Returns:
        the dataloader object
    """

    n_cpus = 4 if not hasattr(os, 'sched_getaffinity') else len(os.sched_getaffinity(0))
    num_workers = min(8, n_cpus) if args.num_workers is None else args.num_workers  # limit to 8 cpus if not specified
    dataloader_args['num_workers'] = num_workers if 'num_workers' not in dataloader_args else dataloader_args['num_workers']
    if not non_verbose:
        logging.info(f'Using {dataloader_args["num_workers"]} workers for the dataloader.')
    if args.seed is not None:
        worker_generator = torch.Generator()
        worker_generator.manual_seed(args.seed)
    else:
        worker_generator = None
    dataloader_args['generator'] = worker_generator if 'generator' not in dataloader_args else dataloader_args['generator']
    init_fn = partial(worker_init_fn, num_workers=num_workers, seed=args.seed) if args.seed is not None else None
    dataloader_args['worker_init_fn'] = init_fn if 'worker_init_fn' not in dataloader_args else dataloader_args['worker_init_fn']

    return DataLoader(dataset, **dataloader_args)
