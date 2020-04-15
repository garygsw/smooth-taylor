import math
import torch
import numpy as np
from torch.utils.data import DataLoader
from scipy.integrate import simps
from skimage.transform import pyramid_gaussian

from .analyzer import integrated_gradients, smooth_integrated_gradients
from .constants import *


def heatmap_normalize(heatmap, percentile=99):
    heatmap = np.abs(heatmap)
    vmin = np.min(heatmap)
    vmax = np.percentile(heatmap, percentile)
    heatmap = (heatmap - vmin) / (vmax - vmin)
    return np.clip(heatmap, 0, 1)


def average_total_variation(img, norm=1):
    """
    Args:
        img (array): image array with the shape (H, W)
    """
    img = heatmap_normalize(img)
    total_pixels = img.shape[0] * img.shape[1]
    x_diff = np.abs(img[:,1:] - img[:,:-1])
    y_diff = np.abs(img[1:,:] - img[:-1,:])
    if norm > 1:
        x_diff = np.power(x_diff, norm)
        y_diff = np.power(y_diff, norm)
        total = np.power(np.sum(x_diff), 1. / norm) + np.power(np.sum(y_diff), 1. / norm)
    else:
        total = np.sum(x_diff) + np.sum(y_diff)
    return total / total_pixels


def compute_autvc(img_input, model, batch_size, transform,
                  analyzer, explained_class,
                  downscale=1.5, min_size=(30, 30), lp_norm=1,
                  baseline_type=None, num_noise=None, steps=None,
                  noise_scale=None, num_roots=None):
    # Compute attribution map
    if analyzer == 'smooth-taylor':
        assert(noise_scale)
        assert(num_roots)
        attributions = smooth_integrated_gradients(
            inputs=img_input,
            model=model,
            batch_size=batch_size,
            noise_scale=noise_scale,
            num_roots=num_roots,
            explained_class=explained_class
        )
    elif analyzer == 'ig':
        assert(steps)
        assert(baseline_type)
        if baseline_type == 'zero':
            baseline = torch.zeros_like(img_input)
            baseline = transform(baseline)
            attributions = integrated_gradients(
                inputs=img_input,
                model=model,
                batch_size=batch_size,
                baseline=baseline,
                explained_class=explained_class,
                steps=steps
            )
        if baseline_type == 'noise':
            assert(num_noise)
            all_igs = []
            for _ in range(num_noise):
                baseline = torch.rand_like(img_input)  # uniform between 0 and 1
                baseline = transform(baseline)
                ig = integrated_gradients(
                    inputs=img_input,
                    model=model,
                    batch_size=batch_size,
                    baseline=baseline,
                    explained_class=explained_class,
                    steps=steps
                )
                all_igs.append(ig)
            attributions = np.mean(np.array(all_igs), axis=0)
    else:
        print('Invalid analyzer:', analyzer)
        exit()

    # Create heatmap by summing across channels
    heatmap = np.sum(attributions, axis=0)

    # Compute multi-scaled averaged TVs
    all_atvs = []
    for (i, resized) in enumerate(pyramid_gaussian(heatmap, downscale=downscale)):
        # if the image is too small, break from the loop
        if resized.shape[0] < min_size[0] or resized.shape[1] < min_size[1]:
            break
        all_atvs.append(average_total_variation(resized, norm=lp_norm))
    all_atvs = np.array(all_atvs)
    autvc = simps(all_atvs, dx=1)
    return heatmap, all_atvs, autvc


def compute_aupc(img_input, model, batch_size, transform,
                 analyzer, explained_class, input_score,
                 kernel_size, draw_mode, num_regions, num_perturbs,
                 baseline_type=None, num_noise=None, steps=None,
                 noise_scale=None, num_roots=None):
    # Compute attribution map
    if analyzer == 'smooth-taylor':
        assert(noise_scale)
        assert(num_roots)
        attributions = smooth_integrated_gradients(
            inputs=img_input,
            model=model,
            batch_size=batch_size,
            noise_scale=noise_scale,
            num_roots=num_roots,
            explained_class=explained_class
        )
    elif analyzer == 'ig':
        assert(steps)
        assert(baseline_type)
        if baseline_type == 'zero':
            baseline = torch.zeros_like(img_input)
            baseline = transform(baseline)
            attributions = integrated_gradients(
                inputs=img_input,
                model=model,
                batch_size=batch_size,
                baseline=baseline,
                explained_class=explained_class,
                steps=steps
            )
        if baseline_type == 'noise':
            assert(num_noise)
            all_igs = []
            for _ in range(num_noise):
                baseline = torch.rand_like(img_input)  # uniform between 0 and 1
                baseline = transform(baseline)
                ig = integrated_gradients(
                    inputs=img_input,
                    model=model,
                    batch_size=batch_size,
                    baseline=baseline,
                    explained_class=explained_class,
                    steps=steps
                )
                all_igs.append(ig)
            attributions = np.mean(np.array(all_igs), axis=0)
    else:
        print('Invalid analyzer:', analyzer)
        exit()

    # Create heatmap by summing across channels
    heatmap = np.sum(attributions, axis=0)

    # Perform perturbation
    perturb_scores = compute_perturbations(
        img_input=img_input,
        model=model,
        batch_size=batch_size,
        explained_class=explained_class,
        heatmap=heatmap,
        kernel_size=kernel_size,
        draw_mode=draw_mode,
        num_regions=num_regions,
        num_perturbs=num_perturbs
    )

    # Prepare the perturb scores
    perturb_scores = [input_score] + perturb_scores   # add original score
    perturb_scores = [x / math.fabs(input_score) for x in perturb_scores]  # normalize into between 0 and 1
    aupc = simps(perturb_scores, dx=1)
    return heatmap, perturb_scores, aupc


def compute_perturbations(img_input, model, batch_size, explained_class, heatmap,
                          kernel_size, draw_mode, num_regions, num_perturbs):
    """
    Args:
        img_input (array): original image
        model (PyTorch model): pretrained model
        batch_size (int): batch size to run per epoch
        explained_class (int): index of the class to be explained
        heatmap (array): attribution heatmap for the image
        kernel_size (int): size of the window of each perturbation
        draw_mode (int): perturb draw mode: 0 - uniform; 1 - gaussian according to image stats
        num_regions (int): number of regions to perturbate
        num_perturbs (int): number of random perturbations to evaluate
    """
    # Find average pooling with kernel
    img_h, img_w = heatmap.shape
    avg_values = -np.inf * np.ones(img_h * img_w)

    for h in range(img_h - kernel_size):
        for w in range(img_w - kernel_size):
            avg_values[h+w*img_h] = np.mean(np.abs(heatmap[h:h+kernel_size, w:w+kernel_size]))
    most_relevant_idxs = np.argsort(-avg_values)   # most relevant first

    # Load original image
    img = INVERSE_TRANSFORM(img_input)  # TODO: maybe take transform as input?
    if draw_mode == 1:   # from gaussian according to img stats
        # compute mean and std dev. from each channel
        channel_stats = np.zeros((2,3))
        for c in range(3):
            channel_stats[0, c]= np.mean(img[c,:,:])
            channel_stats[1, c]= np.std(img[c,:,:])
            if np.isnan(channel_stats[1, c]):
                channel_stats[1, c] = 1e-3

    # Initialize perturb scores
    perturb_scores = []

    # Select the most relevant point to perform the perturbation
    bad_idxs = set()
    for region_idx in range(num_regions):
        found = False
        for i, kernel_idx in enumerate(most_relevant_idxs):
            if kernel_idx not in bad_idxs:
                # get coordinates of point
                width = int(math.floor(kernel_idx / img_h))
                height = int(kernel_idx - width * img_h)

                # ignore if the point is beyond the boundaries of kernel
                if (img_h - height) <= kernel_size or (img_w- width) <= kernel_size:
                    continue

                # mark overlapping neighboring points as not to use in bad_idxs
                for h in range(-kernel_size + 1, kernel_size):
                    for w in range(-kernel_size + 1, kernel_size):
                        bad_idxs.add((height + h) + (width + w) * img_h)
                
                found=True
                break
        if not found:  # no more useful points
            break

        # Prepare perturbation copies
        perturbs = []
        perturbs_imgs = torch.stack([torch.zeros_like(img_input) for _ in range(num_perturbs)])

        # Compute the perturbation
        for i in range(num_perturbs):
            if draw_mode == 0:   # uniform
                # draw randomly from uniform distribution, space is 0,255
                perturb = np.random.uniform(low=0, high=255, size=(3, kernel_size, kernel_size))
            elif draw_mode == 1:   # from gaussian according to img stats
                perturb = np.zeros((3, kernel_size, kernel_size))
                for c in range(3):
                    perturb[c] = np.random.normal(
                        loc=channel_stats[0, c],    # mean
                        scale=channel_stats[1, c],  # std dev
                        size=(kernel_size, kernel_size)
                    )
                # Ensure perturb does not exceed bounds
                perturb = np.maximum(perturb, np.zeros_like(perturb))       # element-wise max(p, 0)
                perturb = np.minimum(perturb, 255 * np.ones_like(perturb))  # element-wise min(p, 255)
            else:
                print('Invalid perturb draw mode')
                exit()
            
            # Apply the perturbation to the current point
            perturbs_imgs[i] = img
            perturb = perturb / 255.  # normalize to 0 to 1
            perturbs_imgs[i, :, height:height+kernel_size, width:width+kernel_size] = torch.Tensor(perturb)
            perturbs_imgs[i] = NORMALIZE_TRANSFORM(perturbs_imgs[i])   # TODO: take transform as input?
            perturbs.append(perturb)
            
        # Perform the classification for the perturbations
        perturb_dataset = torch.utils.data.dataset.TensorDataset(perturbs_imgs)
        data_loader = DataLoader(perturb_dataset, batch_size=batch_size, shuffle=False)
        actual_perturb_idx, mean_score = classify_perturbations(data_loader, model, explained_class)

        # Apply the actual perturbation on image
        actual_perturb = perturbs[actual_perturb_idx]
        img[:, height:height+kernel_size, width:width+kernel_size] = torch.Tensor(actual_perturb)

        # Save the mean scores of the perturbations
        perturb_scores.append(mean_score)
    return perturb_scores


def classify_perturbations(data_loader, model, explained_class):
    """
    Args:
        data_loader (DataLoader): object that loads perturbated images
        model (model): pre-trained PyTorch model
        explained_class (int): index of the class to be explained
    """
    # Find the scores for all perturbation candidates
    all_scores = []
    for sample_batch in data_loader:
        inputs = sample_batch[0]
        inputs = inputs.to(DEVICE)
        with torch.no_grad():
            out = model(inputs)
            scores = out[:, explained_class].cpu().numpy()
            all_scores.append(scores)
    all_scores = np.array(all_scores)
    all_scores = np.concatenate(all_scores)

    # Find the index of the perturbation closest to the median
    mean_score = np.mean(all_scores)
    median = np.median(all_scores)
    best_idx = 0
    best_val = np.inf
    for i in range(len(all_scores)):   # for every perturbation candidate
        value = math.fabs(all_scores[i] - median)  # absolute distance to median
        if value < best_val:
            best_val = value
            best_idx = i
    return best_idx, mean_score
