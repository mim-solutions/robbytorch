"""Functions for running and evaluating models."""
from contextlib import contextmanager
import gc
from typing import Callable, Dict, Iterable, Optional, Tuple, TypeVar, Union
import warnings

import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .utils import Timer, map_structure, get_device


T = TypeVar('T')

# A callback that given (model, inputs, labels) outputs transformed (inputs, labels).
DataTransform = Callable[[nn.Module, Tensor, Tensor], Tuple[Tensor, Tensor]]


def warn_if_not_cuda(module: nn.Module) -> None:
    for p in module.parameters():
        if not str(p.device).startswith("cuda"):
            warnings.warn("Module not on CUDA.", stacklevel=2)
            break


def tensor_to(
    tensor: torch.Tensor,
    where_to: Union[torch.nn.Module, torch.Tensor, torch.device],
    non_blocking=True,
    copy=False,
):
    """Make tensor compatible with `where_to` (eg. move to the where_to.device).

    In the future this could adjust other things like float precision.
    """
    return tensor.to(get_device(where_to), non_blocking=non_blocking, copy=copy)


def tensors_to(
    tensors: Union[DataLoader, Iterable[torch.Tensor]],
    where_to: Union[torch.nn.Module, torch.Tensor, torch.device],
    non_blocking=True,
    copy=False,
):
    """Make tensors compatible with `where_to` (eg. move to the where_to.device)."""
    for t in tensors:
        if isinstance(t, torch.Tensor):
            yield tensor_to(t, where_to, non_blocking=non_blocking, copy=copy)
        else:
            yield [tensor_to(tt, where_to, non_blocking=non_blocking, copy=copy) for tt in t]


def structure_to(
    structure,
    where_to: Union[torch.nn.Module, torch.Tensor, torch.device],
    non_blocking=True,
    copy=False,
):
    """Make structure compatible with `where_to` (eg. move to the where_to.device)."""
    return map_structure(structure, lambda t: tensor_to(t, where_to) if isinstance(t, torch.Tensor) else t)


def clean(cuda: bool = True) -> None:
    """Clean-up interruped tqdm instances and garbage-collect CUDA cache."""
    getattr(tqdm, "_instances", {}).clear()
    if cuda:
        torch.cuda.empty_cache()
    gc.collect()
    print(end="", flush=True)


@contextmanager
def interruptible():
    """Context manager from which keyboard interrupts exit cleanly.

    Instead of the default backtrace, we catch the exception and just print "KeyboardInterrupt".
    The context also always calls `clean(cuda=False)` on exit.
    """
    try:
        yield
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    finally:
        clean(cuda=False)


@contextmanager
def train_mode(module: nn.Module):
    """Context which turns on training mode, and returns to original mode on exit."""
    was_training = module.training
    module.train()
    try:
        yield
    finally:
        module.train(was_training)


@contextmanager
def eval_mode(module: nn.Module):
    """Context which turns on eval mode, and returns to original mode on exit."""
    was_training = module.training
    module.eval()
    try:
        yield
    finally:
        module.train(was_training)
