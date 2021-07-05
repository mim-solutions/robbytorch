import torch
import torchvision
import numpy as np
import pandas as pd
import pathlib


def download_cute_dataset(root=None):
    """Downloads torchvision.datasets.Caltech101 into root_dir (default torch.hub.get_dir()).
    
    Returns pd.Dataframe with metadata for CuteDataset.
    
    Cuteness column is randomized in a small degree.
    """
    map_cuteness = {
        'lobster': 0.1,
        'beaver': 0.25,
        'pigeon': 0.5,
        'llama': 0.75,
        'hedgehog': 0.9,
    }
    name_to_label = {k: i for i, k in enumerate(map_cuteness.keys())}

    root = root or torch.hub.get_dir()
    root = pathlib.Path(root).expanduser()

    # download Caltech Dataset
    torchvision.datasets.Caltech101(root, download=True)
    
    files = [d_path.relative_to(root) for d_path in root.rglob("*.jpg")]
    files = [f for f in files if f.parent.name in map_cuteness] # only 5 labels
    
    file_names = [f.name for f in files]
    label_names = [f.parent.name for f in files]
    labels = [name_to_label[ln] for ln in label_names]

    base_cuteness = torch.tensor([map_cuteness[f.parent.name] for f in files])
    rand = (torch.randn(len(files))/10 + 1) # I'm sorry that it's random ;(
    cuteness = np.array((rand * base_cuteness).clip(0,1))

    df = pd.DataFrame(data=zip(file_names, label_names, labels, cuteness), columns=["file_name", "label_name", "label", "cuteness"])
    
    return df, root / 'caltech101/101_ObjectCategories'