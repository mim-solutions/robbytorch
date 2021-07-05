import torch
from torchvision import transforms
from torchvision.datasets.folder import default_loader
import pathlib
from typing import Mapping, Optional, Callable, Union, Any

from ..utils import SizedIterable, PathLike, TensorLike



class DictDataset(torch.utils.data.Dataset):
    """Base helper class for creating robust Datasets
    
    Args:
        root: Root directory containing your data
        metadata: Any SizedIterable storing metadata for your data, could be a pandas.DataFrame etc.
        transform: If given, takes a single ndarray datapoint and returns a transformed torch tensor.
            Default: `torchvision.transforms.ToTensor`.
    """
    
    def __init__(self, root: PathLike, metadata: SizedIterable, transform: Optional[Callable[[TensorLike], torch.Tensor]] = None):
        self.root = pathlib.Path(root).expanduser()
        self.metadata = metadata
        self.transform = transform

    def load_image(self, file_name: PathLike) -> TensorLike:
        """Helper method for loading images (they have to be inside self.root directory)"""
        return default_loader(self.root / file_name)

    def load_data(self, idx: int) -> TensorLike:
        """Should return a dataitem for a given integer key.
        
        You don't need to apply self.transform in this function (it will be applied later in self.__getitem__). 
        Hence the returned dataitem need not be a tensor.
        """
        raise NotImplementedError()
    
    def load_target_dict(self, idx: int) -> Mapping[str, Any]:
        """Should return a dict of Any data. It will be merged with {"data": dataitem} and returned by self.__getitem__.

        Here you should return the target value for your respective dataitem (for the given `idx`) and any auxiliary
        metadata that might be usefull in computing the (augmented) loss function
        """
        raise NotImplementedError()

    def __getitem__(self, idx: int) -> Mapping[str, torch.Tensor]:
        """Returns a {"data": data, **self.load_target_dict(idx)} dataitem for a given integer key. 
            
        Every value has to be a torch.Tensor.

        Later torch.utils.data.DataLoader collates this dicts into a dict o batched tensors.
        """
        data = self.load_data(idx)
        
        if self.transform:
            data = self.transform(data)
        else:
            data = transforms.ToTensor()(data)
        
        return {"data": data, **self.load_target_dict(idx)}        
        
    def __len__(self) -> int:
        return len(self.metadata)