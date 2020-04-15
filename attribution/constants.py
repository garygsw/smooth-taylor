import os
import numpy as np
import torch
from torchvision import models, transforms


MODELS = {'densenet121': models.densenet121,
          'resnet152': models.resnet152}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ANALYZERS = ['grad', 'smooth-grad', 'smooth-taylor', 'ig', 'lrp']
IG_BASELINES = ['zero', 'noise']


# ImageNet transform constants
DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize(256),       # resize image to 256X256 pixels
    transforms.CenterCrop(224),   # crop the image to 224X224 pixels about the center
    transforms.ToTensor(),        # convert the image to PyTorch Tensor data type
    transforms.Normalize(         # Normalize by setting the mean and s.d. to specified values
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

NORMALIZE_TRANSFORM = transforms.Compose([
    transforms.Normalize(         # Normalize by setting the mean and s.d. to specified values
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

RESIZE_TRANSFORM = transforms.Compose([
    transforms.Resize(256),       # resize image to 256X256 pixels
    transforms.CenterCrop(224),   # crop the image to 224X224 pixels about the center
])

INVERSE_TRANSFORM = transforms.Compose([
    transforms.Normalize(         # Normalize by setting the mean and s.d. to specified values
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
])