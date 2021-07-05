"""
Example transforms
"""

import torch
import torchvision.transforms as transforms


# Data Augmentation defaults
TRAIN_TRANSFORMS_DEFAULT = lambda size: transforms.Compose([
            transforms.RandomCrop(size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.25,.25,.25),
            transforms.RandomRotation(2),
            transforms.ToTensor(),
        ])
"""
Generic training data transform, given image side length does random cropping,
flipping, color jitter, and rotation. Called as, for example,
:meth:`robustness.data_augmentation.TRAIN_TRANSFORMS_DEFAULT(32)` for CIFAR-10.
"""

TEST_TRANSFORMS_DEFAULT = lambda size: transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor()
    ])
"""
Generic test data transform (no augmentation) to complement
:meth:`robustness.data_augmentation.TEST_TRANSFORMS_DEFAULT`, takes in an image
side length.
"""

# for transfer of ImageNet models
TRAIN_TRANSFORMS_TRANSFER = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

TEST_TRANSFORMS_TRANSFER = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])



class ReshapeTransform:
    
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)



class TransposeTransform:
    
    def __init__(self, dim1, dim2):
        self.dim1 = dim1
        self.dim2 = dim2

    def __call__(self, img):
        return img.transpose(self.dim1, self.dim2)


class Ensure3ChannelTransform:
    # FIXME - this class is hacky
    # Some images have more then 3 channels (thermal channel etc.)
    # or 1 channel (greyscale)

    def __call__(self, img):
        img = img[:3]
        if img.shape[0] < 3:
            img = img[0].unsqueeze(0).expand(3, -1, -1)

        return img 
