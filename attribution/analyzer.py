import torch
import numpy as np
from torch.utils.data import DataLoader

from .constants import *


def integrated_gradients(inputs, model, batch_size, baseline, explained_class, steps=50):
    """
    Args:
        inputs (array): original inputs
        model (PyTorch model): pretrained model
        batch_size (int): batch size to run per epoch
        baseline (array): reference point
        explained_class (int): index of the class to be explained
        steps (int, optional): number of steps along path
    """
    # Scale the input with k/m progressive steps multiplier
    diff = inputs - baseline
    scaled_inputs = [baseline + (float(i) / steps) *  diff for i in range(steps + 1)]
    scaled_inputs = torch.stack(scaled_inputs, dim=0)   # shape: (k + 1, C, H, W)
    scaled_dataset = torch.utils.data.dataset.TensorDataset(scaled_inputs)
    scaled_loader = DataLoader(scaled_dataset, batch_size=batch_size, shuffle=False)

    # Get the gradients along this path
    gradients = get_gradients(scaled_loader, model, explained_class=explained_class)
    avg_grads = np.average(gradients[1:], axis=0)  # from step 1 onwards
    
    integrated_grad = diff.numpy() * avg_grads
    return integrated_grad


def smooth_taylor(inputs, model, batch_size, noise_scale, num_roots, explained_class, percent=False):
    """
    Args:
        inputs (array): original inputs
        model (PyTorch model): pretrained model
        batch_size (int): batch size to run per epoch
        noise_scale (float): scale to noise the inputs
        num_roots (int): number of noised inputs to generate
        explained_class (int): index of the class to be explained
        percent (bool, optional): use noise scale percentage
    """
    # Generate roots dataset based on the original inputs with additive noise
    roots = torch.stack([torch.zeros_like(inputs) for _ in range(num_roots)])
    if percent:
        noise_scale = noise_scale * (np.max(inputs.numpy()) - np.min(inputs.numpy()))
    for i in range(num_roots):
        roots[i] = inputs + noise_scale * torch.randn_like(inputs)
    roots_dataset = torch.utils.data.dataset.TensorDataset(roots)
    roots_data_loader = DataLoader(roots_dataset, batch_size=batch_size, shuffle=False)

    # Compute the gradients w.r.t. explained class for roots
    gradients = get_gradients(roots_data_loader, model, explained_class=explained_class)

    # Compute SmoothTaylor with contribution from roots
    attributions = np.mean([(inputs - roots_dataset[i][0]).numpy() * gradients[i]
                             for i in range(num_roots)], axis=0)
    return attributions


def get_gradients(data_loader, model, explained_class):
    """
    Args:
        data_loader (DataLoader): object that loads images data
        model (model): pre-trained PyTorch model
        explained_class: index of the class to be explained
    """
    gradients = []
    for sample_batch in data_loader:
        inputs = sample_batch[0]
        inputs = inputs.to(DEVICE)
        inputs.requires_grad = True

        # Perform the backpropagation for the explained class
        out = model(inputs)
        model.zero_grad()
        torch.sum(out[:,explained_class]).backward()
        with torch.no_grad():
            gradient = inputs.grad.detach().cpu().numpy()  # retrieve the input gradients
            gradients.append(gradient)
    gradients = np.array(gradients)
    gradients = np.concatenate(gradients)
    return gradients
