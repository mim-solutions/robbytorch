import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pathlib, shutil, os

from typing import overload, Callable, Dict, Generic, Iterable, Iterator, List, Mapping, Sequence, \
                   Tuple, TypeVar, Union

try:
    from typing import Protocol
except ImportError:  # Workaround for Python < 3.8.
    Protocol = Generic

from .input_transforms import PGD
from . import visualization


PathLike = Union[str, os.PathLike]
TensorLike = Union[np.ndarray, torch.Tensor]


_T_co = TypeVar("_T_co", covariant=True)


class SizedIterable(Iterable[_T_co], Protocol[_T_co]):
    """A SizedIterable is any Iterable type that supports `len`."""

    def __len__(self) -> int:
        ...


_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)
_S = TypeVar("_S")
_V = TypeVar("_V")


class SizedIterable(Iterable[_T_co], Protocol[_T_co]):
    """A SizedIterable is any Iterable type that supports `len`."""

    def __len__(self) -> int:
        ...


class map_iterable(SizedIterable[_S], Generic[_S]):
    """Apply a function to every item of an iterable, preserving `__len__`.

    `map_iterable(f, it)` works just as `map(f, it)`,
    except it returns a `SizedIterable` instead of an iterator:
    you can call `len()` on it and you can call `iter()` repeatedly on it.
    """

    def __init__(self, func: Callable[[_T], _S], iterable: SizedIterable[_T]):
        self.func = func
        self.iterable = iterable

    def __iter__(self) -> Iterator[_S]:
        for item in self.iterable:
            yield self.func(item)

    def __len__(self) -> int:
        return len(self.iterable)


# map_structure is hard to type properly, but here we handle the most common cases.
@overload
def map_structure(data: _V, f: Callable[[_T], _T]) -> _V: ...
@overload
def map_structure(data: Tuple[_T, ...], f: Callable[[_T], _S]) -> Tuple[_S, ...]: ...
@overload
def map_structure(data: Dict[_V, _T], f: Callable[[_T], _S]) -> Dict[_V, _S]: ...
@overload
def map_structure(data: Mapping[_V, _T], f: Callable[[_T], _S]) -> Mapping[_V, _S]: ...
@overload
def map_structure(data: List[_T], f: Callable[[_T], _S]) -> List[_S]: ...
@overload
def map_structure(data: Sequence[_T], f: Callable[[_T], _S]) -> Sequence[_S]: ...


def map_structure(data, f: Callable):
    """Apply a function to every item in a structure, recursing into lists, dicts, etc.

    For example `map_structure((a, b), f)` returns `(f(a), f(b))` when a, b are tensors.

    We recurse into `Mapping` and `Sequence` types.
    We do not recurse into `Iterable` types in general: for example `f` is applied
    to a `torch.Tensor` as a whole, not to each element individually.

    This mimics how pytorch goes through dataitems when collating them into batches,
    see `torch.utils.data.dataloader.default_collate`
    or `torch.utils.data._utils.pin_memory.pin_memory`.
    """
    if isinstance(data, tuple):  # tuple or subtypes like namedtuple
        return type(data)(*(map_structure(v, f) for v in data))
    elif isinstance(data, dict):  # dict or subtypes like OrderedDict
        return type(data)(**{k: map_structure(v, f) for k, v in data.items()})
    elif isinstance(data, Mapping):
        return {k: map_structure(v, f) for k, v in data.items()}
    elif isinstance(data, List) or isinstance(data, Tuple):
        return [map_structure(v, f) for v in data]
    else:
        return f(data)


def show_structure(st):
    """Returns structure resembling given structure but with simple repr"""
    ret = {}
    for k, v in st.items():
        if isinstance(v, dict):
            ret[k] = show_structure(st[k])
        elif isinstance(v, torch.Tensor):
            ret[k] = v.shape
        elif isinstance(v, list):
            ret[k] = len(v)
    return ret



class Timer():
    def __init__(self) -> None:
        self.started = self.time()

    def time(self) -> float:
        return time.monotonic()

    def elapsed(self) -> float:
        return self.time() - self.started


class DefaultDict(dict):
    """Same as collections.defaultdict but you can pass arguments to the factory function."""

    def __init__(self, factory):
        super().__init__
        self.factory = factory

    def __missing__(self, key):
       res = self[key] = self.factory(key)
       return res


class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_device(what: Union[torch.nn.Module, torch.Tensor, torch.device]):
    """Returns the device of the first tensor parameter in `what`
    
    WARNING: tensor's devices in the model might be inconsistent.
    """
    while isinstance(what, torch.nn.Module):
        what = next(iter(what.parameters()))
    if isinstance(what, torch.Tensor):
        what = what.device

    return what


def mkdir_and_preserve_group(path: PathLike) -> str:
    """Mkdir all ancestors and set the same group owner as the first existing ancestor."""
    path = pathlib.Path(path)
    ancestor = path
    while not ancestor.exists():
        ancestor = ancestor.parent
    group = ancestor.group()
    path.mkdir(parents=True, exist_ok=True)
    ancestor = path
    while not ancestor.group() == group:
        shutil.chown(ancestor, group=group)
        ancestor = ancestor.parent
    return group


def flatten_dict(dictionary, sep='/'):
    """Flatten nested dictionaries, compressing keys"""
    if not dictionary:
        return {}
    df = pd.json_normalize(dictionary, sep=sep)
    return df.to_dict(orient='records')[0]


def transfer_model(model, num_classes=2):
    model.fc = nn.Linear(model.fc.in_features, num_classes)


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def visualize_gradient(t):
    '''
    Visualize gradients of model. To transform gradient to image range [0, 1], we 
    subtract the mean, divide by 2*3 standard deviations, and then clip.
    
    Args:
        t (tensor): input tensor (usually gradients)
    '''  
    mt = torch.mean(t, dim=[2, 3], keepdim=True).expand_as(t)
    st = torch.std(t, dim=[2, 3], keepdim=True).expand_as(t)
    return torch.clamp((t - mt) / (2*3 * st) + 0.5, 0, 1)


def get_accuracy(logits, target):
    pred = logits.argmax(dim=1)
    accuracy = (pred == target).sum().item() / len(target) * 100
    return accuracy


def memory_summary(device = 'cuda'):
    """pass `cuda:1` for summary for cuda:1"""
    return torch.cuda.memory_summary(device)