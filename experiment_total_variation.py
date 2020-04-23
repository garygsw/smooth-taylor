import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
from torch.utils.data import DataLoader

from attribution.eval import compute_multiscaled_atv
from attribution.dataset import ImageNetValDataset
from attribution.constants import *


def parse_args():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Generate multi-scaled average total variation based on a set of heatmaps')
    parser.add_argument('-m', '--model_name', type=str, help='name of the model used to classify the noised images')
    parser.add_argument('-a', '--analyzer', type=str, help='analyzer that compute the heatmaps')
    parser.add_argument('-z', '--baseline', type=str, help='baseline type to compute IG', default='zero')
    parser.add_argument('-n', '--num_noise', type=int, help='number of noise baselines used by IG', default=1)
    parser.add_argument('-s', '--noise_scale', type=float, help='magnitude of the noise scale used to noise the image', default=5e-1)
    parser.add_argument('-p', '--noise_percent', action='store_true', help='use proportion of image as noise scale')
    parser.add_argument('-an', '--adaptive_noise', action='store_true', help='use adaptive noise')
    parser.add_argument('-af', '--adaptive_function', type=str, help='objective function used', default='aupc')
    parser.add_argument('-r', '--num_roots', type=int, help='number of noised images used', default=150)
    parser.add_argument('-ds', '--downscale', type=float, help='factor to downscale heatmap', default=1.5)
    parser.add_argument('-wms', '--width_min_size', type=int, help='minimum width dimension for downscale', default=30)
    parser.add_argument('-hms', '--height_min_size', type=int, help='minimum height dimension for downscale', default=30)
    parser.add_argument('-lp', '--lp_norm', type=int, help='norm to use to calculate TV', default=1)
    parser.add_argument('-i', '--num_image', type=int, help='number of image data to use from the first', default=1000)
    parser.add_argument('-o', '--overwrite', action='store_true', help='overwrite the output')
    args = parser.parse_args()
    if args.model_name not in MODELS:
        print('Invalid model name:', args.model_name)
        exit()
    if args.analyzer not in ANALYZERS:
        print('Invalid analyzer:', args.analyzer)
        exit()
    if args.baseline not in IG_BASELINES:
        print('Invalid IG baseline name:', args.baseline)
        exit()
    if args.adaptive_function not in ['aupc', 'autvc']:
        print('Invalid adaptive objective function:', args.adaptive_function)
        exit()
    return args


def run_total_variation_evaluation(dataset, model_name, analyzer,
                                   noise_scale=None, num_roots=None,
                                   noise_percent=False, baseline=None, num_noise=None,
                                   downscale=1.5, min_size=(30, 30), lp_norm=1,
                                   adaptive_noise=False, adaptive_function=None, overwrite=False):
    # Prepare input heatmaps directory
    if analyzer == 'smooth-taylor':
        assert(noise_scale)
        assert(num_roots)
        if adaptive_noise:
            assert(adaptive_function)
            noise_folder = 'adaptive/' + adaptive_function
        elif noise_percent:
            noise_folder = str(noise_scale) + '%'
        else:
            noise_folder = '{:.1e}'.format(noise_scale)   # convert to scientific notation
        num_roots_folder = str(num_roots) + 'N'
        heatmap_dir = os.path.join('heatmaps/', analyzer, model_name, noise_folder, num_roots_folder)
    elif analyzer == 'ig':
        assert(baseline)
        if baseline == 'zero':
            heatmap_dir = os.path.join('heatmaps/', analyzer, model_name, baseline)
        elif baseline == 'noise':
            assert(num_noise)
            num_noise_folder = str(num_noise) + 'N'
            heatmap_dir = os.path.join('heatmaps/', analyzer, model_name, baseline, num_noise_folder)
    elif analyzer == 'grad':
        heatmap_dir = os.path.join('heatmaps/', analyzer, model_name)
    elif analyzer == 'smooth-grad':
        assert(num_noise)
        assert(noise_scale)
        noise_folder = str(noise_scale) + '%'
        num_noise_folder = str(num_noise) + 'N'
        heatmap_dir = os.path.join('heatmaps/', analyzer, model_name, noise_folder, num_noise_folder)
    if not os.path.exists(heatmap_dir):
        print('Heatmaps folders missing:', heatmap_dir)
        exit()

    # Prepare output directory
    output_dir = os.path.join('atv/', analyzer, model_name)
    if analyzer == 'smooth-taylor':
        output_dir = os.path.join(output_dir, noise_folder, num_roots_folder)
    elif analyzer == 'ig':
        if baseline == 'zero':
            output_dir = os.path.join(output_dir, baseline)
        elif baseline == 'noise':
            output_dir = os.path.join(output_dir, baseline, num_noise_folder)
    elif analyzer == 'smooth-grad':
        output_dir = os.path.join(output_dir, noise_folder, num_noise_folder)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Go through each image and compute ATV scores
    all_atv_scores = []
    for img_idx, img_filepath in enumerate(tqdm(dataset.img_filepaths, desc='Image')):
        # Check if ATV scores exists
        img_filename = os.path.basename(img_filepath)
        atv_filename = 'atv_scores_{0}ds_{1}ms_{2}l_{3}.npy'.format(
            downscale, min_size, lp_norm, img_filename
        )
        atv_outpath = os.path.join(output_dir, atv_filename)
        if not overwrite and os.path.exists(atv_outpath):  # ignore if done
            print(img_filename, 'already has ATV scores computed')
            all_atv_scores.append(np.load(atv_outpath))
            continue

        # Load the image heatmap
        heatmap_filepath = os.path.join(heatmap_dir, img_filename + '_hm.npy')
        if not os.path.exists(heatmap_filepath):
            print('Heatmaps not found for:', heatmap_filepath)
            continue
        heatmap = np.load(heatmap_filepath)

        multi_scaled_ATV = compute_multiscaled_atv(
            heatmap=heatmap,
            downscale=downscale,
            min_size=min_size,
            lp_norm=lp_norm
        )

        # Update all ATV scores and save individual scores
        all_atv_scores.append(multi_scaled_ATV)
        np.save(atv_outpath, multi_scaled_ATV)
    
    # Save all ATV scores
    out_filename = 'all_ATV_scores_{0}ds_{1}ms_{2}l.npy'.format(
        downscale, min_size, lp_norm
    )
    outpath = os.path.join(output_dir, out_filename)
    all_atv_scores = np.array(all_atv_scores)
    np.save(outpath, all_atv_scores)


if __name__ == "__main__":
    args = parse_args()

    from datetime import datetime, timedelta
    start_time = datetime.now()

    # Load the dataset
    dataset = ImageNetValDataset(
        root_dir='data/images',
        label_dir='data/annotations',
        synset_filepath='rsc/synset_words.txt',
        max_num=args.num_image
    )

    # Perform total variation experiment
    run_total_variation_evaluation(
        dataset=dataset,
        model_name=args.model_name,
        analyzer=args.analyzer,
        noise_scale=args.noise_scale,
        noise_percent=args.noise_percent,
        num_roots=args.num_roots,
        baseline=args.baseline,
        num_noise=args.num_noise,
        downscale=args.downscale,
        min_size=(args.height_min_size, args.width_min_size),
        lp_norm=args.lp_norm,
        adaptive_noise=args.adaptive_noise,
        adaptive_function=args.adaptive_function,
        overwrite=args.overwrite
    )

    end_time = datetime.now()
    elapsed_seconds = int((end_time - start_time).total_seconds())
    print('Start time:', start_time.strftime('%d %b %Y %H:%M:%S'))
    print('End time:', end_time.strftime('%d %b %Y %H:%M:%S'))
    print('Elapsed time:', timedelta(seconds=elapsed_seconds))