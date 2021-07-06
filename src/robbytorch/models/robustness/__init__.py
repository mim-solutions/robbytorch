import pathlib
import sys
from typing import Dict, Optional, Union#, Literal
import warnings

import dill
import torch
import torch.nn
import torch.optim
import torchvision.models
import torch.hub

from . import resnet as robustness_resnet
# TODO remove this dependency and use torchvision.models.
# It's basically an exact copy of torchvision.models.resnet18, except in the following.
# - The ReLU in the last BasicBlock of the last layer (before pooling) is optionally
#   disabled or replaced with a fake ReLU whose derivative is always 1.
#   It would probably be more elegant to hook torchvision.models.resnet18.
# - It exposes "latent features" (activations just before the last fc layer),
#   by changing the arguments, return values and implementation of `forward`.
#   It would be more elegant to just use: ```
#       latent_dim = resnet.fc.in_features
#       resnet.fc = torch.nn.Identity()
#       model = nn.Sequential(OrderedDict(
#           backbone=resnet,
#           head=torch.nn.Linear(latent_dim, len(CLASSES))
#       )).cuda()
#   ```
#   And then when you actually need both latents and logits, use: ```
#       latents = model.backbone(x)
#       logits = model.head(latents)
#   ```
#   Or `latents = model[:-1](x)` in general.

from ...utils import PathLike, mkdir_and_preserve_group


ROBUSTNESS_URL = ("https://robustnessws4285631339.blob.core.windows.net"
                  "/public-models/robust_imagenet/{}?sv=2019-10-10&ss=b&srt=sco&sp=rlx"
                  "&se=2021-10-05T15:06:23Z&st=2020-06-10T07:06:23Z&spr=https"
                  "&sig=Rwwsg9yfcSrbNLvxse%2F32XOy7ERWSLXMz9Ebka4pS20%3D")
# Apparently non-resnet models were trained with original pytorch models, even if the `robustness`
# lirbary provides alternative definitions.
ROBUSTNESS_ARCHITECTURES = {
    "resnet18": robustness_resnet.resnet18,
    "resnet50": robustness_resnet.resnet50,
    "wide_resnet50_2": robustness_resnet.wide_resnet50_2,
    "wide_resnet50_4": robustness_resnet.wide_resnet50_4,
    "densenet": torchvision.models.densenet161,
    "mnasnet": torchvision.models.mnasnet1_0,
    "mobilenet": torchvision.models.mobilenet_v2,
    "resnext50_32x4d": torchvision.models.resnext50_32x4d,
    "shufflenet": torchvision.models.shufflenet_v2_x1_0,
    "vgg16_bn": torchvision.models.vgg16_bn
}
ROBUSTNESS_HASHES = {
    "resnet18_l2_eps0.ckpt": "d762c58693a2ecaf93160060bb2c32493598540a08852fb3068e95e88f032623",
    "resnet18_l2_eps3.ckpt": "b4315784a1a10257b05906aee94d6884678f7b77412d10100d7c05daa959eab0"
}


def get_model_from_robustness(
    arch: str,
    pretraining: Optional[str], # Literal[None, "pytorch", "microsoft"]
    eps: float = 0.0,
    metric: str = "L2",
    device: Union[torch.device, str] = "cpu",
    progress: bool = True,
    cache_dir: Optional[PathLike] = None
) -> torch.nn.Module:
    """Get a model from the `robustness` library, downloading or caching pretrained parameters.

    All pretrained models were pretrained with ImageNet. See and cite:
    - https://github.com/MadryLab/robustness
    - https://github.com/microsoft/robust-models-transfer

    Args:
        arch: architecture name like "resnet18" or "wide_resnet50_4", see ROBUSTNESS_ARCHITECTURES.
        pretraining: None, "pytorch", or "microsoft". Only the latter are available for eps > 0.
        eps: radius of attack the model was pretrained against; 0 means normally pretrained models.
        metric: either "L2" or "Linf", metric of attack the model was pretrained against.
        device: where to load the model's tensors to.
        progress: whether to show a progress bar if downloading.
        cache_dir: directory for the downloaded model parameters, defaults to the "checkpoints"
            subdirectory of `torch.hub.get_dir()`.

    Using None or "pytorch" differs from `torchvision.models.resnet18(pretrained=True)` in that the
    model is the one from the `robustness` library (with the last ReLU deleted etc.),
    only the pretrained parameters values are from pytorch. Moreover, we load the model to `device`.
    """
    if arch not in ROBUSTNESS_ARCHITECTURES:
        raise ValueError(f"Got arch={arch!r}, should be one of {ROBUSTNESS_ARCHITECTURES.keys()}.")
    if eps and pretraining != "microsoft":
        raise ValueError("Incompatible options eps={eps!r} and pretraining={pretraining!r}."
                         " Use pretraining='microsoft' to get robustly pretrained models.")

    if pretraining == "microsoft":
        state_dict = load_state_dict_from_robustness(arch, eps, metric, cache_dir, device, progress)
        model = ROBUSTNESS_ARCHITECTURES[arch]()
        model.load_state_dict(state_dict)
    elif pretraining == "pytorch":
        assert cache_dir is None, "Only the default pytorch cache dir can be used."
        model = ROBUSTNESS_ARCHITECTURES[arch](pretrained=True)
    elif pretraining is None:
        model = ROBUSTNESS_ARCHITECTURES[arch]()
    else:
        raise ValueError("Unknown pretraining, got {pretraining!r},"
                         " should be 'microsoft', 'pytorch', or None.")
    return model.to(device=device)


def load_state_dict_from_robustness(
    arch: str,
    eps: float,
    metric: str = "L2",
    cache_dir: Optional[PathLike] = None,
    device: Union[torch.device, str] = "cpu",
    progress: bool = True
) -> Dict:
    if arch not in ROBUSTNESS_ARCHITECTURES:
        raise ValueError(f"Got arch={arch!r}, should be one of {ROBUSTNESS_ARCHITECTURES.keys()}.")
    if metric.upper() not in ["L2", "Linf"]:
        raise ValueError(f"Got metric={metric!r}, should be one of 'L2', 'Linf'.")

    eps_name = str(int(eps)) if eps.is_integer() else str(eps)
    model_name = f"{arch}_{metric.lower()}_eps{eps_name}.ckpt"

    # Download just as `torch.hub.load_state_dict_from_url`,
    # except that we give hashes separately from the url,
    # and we need to pass `pickle_module=dill`.
    url = ROBUSTNESS_URL.format(model_name)
    hash = ROBUSTNESS_HASHES.get(model_name)
    if hash is None:
        warnings.warn(f"We don't have a sha256 hash stored for {model_name!r},"
                      f" consider adding it to `ROBUSTNESS_HASHES`.")
    if cache_dir is None:
        cache_dir = pathlib.Path(torch.hub.get_dir()) / "checkpoints"
    else:
        cache_dir = pathlib.Path(cache_dir)

    if not cache_dir.exists():
        warnings.warn(f"Saving in `{cache_dir}` but it does not exists. I'm creating the path now.")
        mkdir_and_preserve_group(cache_dir)

    path = cache_dir / model_name
    if not path.exists():
        print(f"Downloading {model_name} from {url} to {cache_dir}", file=sys.stderr)
        try:
            torch.hub.download_url_to_file(url, path, hash_prefix=hash, progress=progress)
        except Exception as e:
            raise Exception("Failed to download model parameters, perhaps"
                            " the URL expired and you need to update `ROBUSTNESS_URL`?") from e
    state_dict = torch.load(path, map_location=device, pickle_module=dill)

    # The `robustness` library wraps each model in layers that we don't use.
    # We have to remove those layers from the state_dict.
    state_dict = state_dict["model"]
    prefix = "module.attacker.model."
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
