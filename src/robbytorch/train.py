import copy
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, List, Dict, Iterable, Callable, Union, Type
from pathlib import Path

from . import utils
from .run import eval_mode, interruptible, tensors_to, train_mode, warn_if_not_cuda, DataTransform


# A callback that given a module returns values to log, e.g. {'loss': ..., 'accuracy': ...}.
Evaluator = Callable[[nn.Module], Dict[str, float]]

# Classes in `torch.optim.lr_scheduler`: they don't have a common base class.
# TODO we could define a Scheduler ABC/protocol.
Scheduler = object

Loss = Callable[[Tensor, Tensor], Tensor]



class Trainer(object):

    def __init__(self, 
        train_loader: DataLoader,
        val_loader: DataLoader,
        forward: DataLoader,
        augment_data: Optional[Callable] = None,
        loss_key: Optional[str] = 'loss'
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.forward = forward
        self.augment_data = augment_data
        self.loss_key = loss_key


    def clone(self,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        forward: Optional[DataLoader] = None,
        augment_data: Optional[Callable] = None,
        loss_key: Optional[str] = 'loss'
    ):
        return self.__class__(
            train_loader=train_loader or self.train_loader,
            val_loader=val_loader or self.val_loader,
            forward=forward or self.forward,
            augment_data=augment_data or self.augment_data,
            loss_key=loss_key or self.loss_key
        )


    def eval_loop(self, model, loader=None):
        loader = loader or self.val_loader
        return self._loop(model, loader=loader)


    def _loop(self, model, optimizer=None, train=False, loader=None, epoch="-"):
        meters = utils.DefaultDict(lambda _: utils.AverageMeter())
        if loader is None:
            loader = self.train_loader if train else self.val_loader
        iterator = tqdm(iter(loader), total=len(loader))
        iterator.set_description(f"Epoch: {epoch}, {'Train' if train else 'VAL'}")

        for dataitem in iterator:
            postfix = {}
            model.train() if train else model.eval()
            data_len = len(dataitem["data"])
            
            if self.augment_data:
                # add more keys to dataitem (i.e. adversarial examples)
                dataitem = self.augment_data(model, dataitem, "train" if train else "eval")

            if train:
                train_result = self.forward(model, dataitem, "train")
                loss = train_result[self.loss_key]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                for k, v in train_result.items():
                    meters[f"_before_update_{k}"].update(_item(v), data_len)
                postfix['_before_update_loss'] = meters[f"_before_update_{self.loss_key}"].avg

            with eval_mode(model), torch.no_grad():
                
                eval_result = self.forward(model, dataitem, "train_eval" if train else "eval")

                for k, v in eval_result.items():
                    meters[k].update(_item(v), data_len)
                postfix['loss'] = meters[self.loss_key].avg

            iterator.set_postfix(refresh=False, **postfix)

        return meters


    def train_model(self, model, 
                optimizer: Union[str, torch.optim.Optimizer, Dict] = "Adam",
                scheduler: Union[None, Scheduler, Dict] = None,
                epochs=10,
                eval_per=10,
                writers=[],
                eval_before_training=False):
        optimizer = get_optimizer(optimizer, model)
        scheduler = get_scheduler(scheduler, optimizer)

        with interruptible():
            if eval_before_training:
                self._loop(model)

            for epoch in range(1, epochs+1):
                meters = self._loop(
                    model,
                    optimizer=optimizer,
                    train=True,
                    epoch=epoch
                )

                if epoch % eval_per == 0 or epoch == epochs:
                    val_meters = self._loop(
                        model, 
                        optimizer=None, 
                        train=False, 
                        epoch=epoch
                    )
                    val_meters = {_prepend_log_key("val", k): v for k, v in val_meters.items()}
                    meters = {**meters, **val_meters}

                if scheduler:
                    scheduler.step()

                meters = {k: v.avg for k, v in meters.items()}
                for writer in writers:
                    writer.log_metrics(meters, epoch, epochs, model)

        return



def get_optimizer(
    spec: Union[str, Dict, torch.optim.Optimizer, Type[torch.optim.Optimizer]], model: nn.Module
) -> torch.optim.Optimizer:
    """Make an optimizer for a model and a given specification.

    The specification should be either:
    - an `torch.optim.Optimizer` instance – we just return it;
    - a subclass of `torch.optim.Optimizer` – we instantiate it with model.parameters();
    - a string giving the name of a class in `torch.optim` like "AdamW" – we use it as above;
    - a dictionary in which:
        - `spec["id"]` is used as the optimizer subclass or name as above,
        - `spec["filter_requires_grad"]` - some optimizers may update parameters that do not require grad (i.e. weight decay). 
            This option is set to filter them out so that they don't get updated. Default: True.
        - `spec["params"]` is either:
            - not given, in which case model.parameters() are used,
            - an iteratable of parameters
            - a list of dictionaries as expected by `torch.optim.Optimizer`,
            - a dictionary of parameter groups:
                - keys like ".foo.bar" are evaluated as `model.foo.bar.parameters()`,
                - values are options for this parameter group,
        - remaining keys are kwargs for the optimizer (default options for all parameter groups).

    Example: ```
        get_optimizer(model, {
            "id": "SGD",
            "params": {
                ".base": {"lr": 1e-3, "momentum": 0}
                ".classifier": {}
            },
            "lr": 1e-2, "momentum": 0.9
        ))
    ```

    Remember to call it on the model _after_ moving it to CUDA, otherwise
    it won't use the same parameters.
    """
    if isinstance(spec, torch.optim.Optimizer):
        return spec
    if isinstance(spec, str):
        spec = {"id": spec}
    if not isinstance(spec, dict):
        raise TypeError(f"Unexpected type: {type(spec)}.")

    spec = copy.copy(spec)  # We don't want to modify the original dict.
    name = spec["id"]
    filter_requires_grad = spec.get("filter_requires_grad", True)
    del spec["id"]
    if "filter_requires_grad" in spec:
        del spec["filter_requires_grad"]
    
    if isinstance(name, str):
        cls = getattr(torch.optim, name, None)
    else:
        cls = name
    if not issubclass(cls, torch.optim.Optimizer):
        raise ValueError(f"No such optimizer in torch.optim: {name}")

    params = spec.get("params", model.parameters())
    if isinstance(params, dict):
        new_params: List[Dict] = []
        for key, options in params.items():
            options = copy.copy(options)
            options["params"] = get_submodule(model, key).parameters()
            if filter_requires_grad:
                options["params"] = filter(lambda p: p.requires_grad, options["params"])
            new_params.append(options)
        params = new_params
    spec["params"] = params

    return cls(**spec)


def get_submodule(module: nn.Module, path: str) -> nn.Module:
    result = module
    names = path.split(".")
    if not names or names[0]:
        raise ValueError(f"Submodule path should start with '.', got: {path!r}")
    for i, name in enumerate(names[1:]):
        result = getattr(result, name, None)
        if not isinstance(result, nn.Module):
            p = ".".join(names[: (i + 1)])
            raise AttributeError(f"No submodule {p} in {module}")
    return result


def get_scheduler(
    spec: Union[None, Scheduler, Dict], optimizer: torch.optim.Optimizer
) -> Optional[Scheduler]:
    if spec is None:
        return None
    if not isinstance(spec, dict):
        return spec  # Assume we got a scheduler instance.

    spec = copy.copy(spec)  # We don't want to modify the original dict.
    name = spec["id"]
    del spec["id"]
    if isinstance(name, str):
        cls = getattr(torch.optim.lr_scheduler, name, None)
    else:
        cls = name
    if not callable(cls):
        raise ValueError(f"No such scheduler in torch.optim.lr_scheduler: {name}")

    return cls(optimizer, **spec)


def _item(v):
    return v.item() if isinstance(v, Tensor) else v


def _prepend_log_key(prefix, key):
    if key.startswith("_") and not prefix.startswith("_"):
        return f"_{prefix}_{key}"
    else:
        return f"{prefix}_{key}"