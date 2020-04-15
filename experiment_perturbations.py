import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
from torch.utils.data import DataLoader

from attribution.eval import compute_perturbations
from attribution.dataset import ImageNetValDataset
from attribution.constants import *


def parse_args():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Generate perturbation scores based on a set of heatmaps')
    parser.add_argument('-m', '--model_name', type=str, help='name of the model used to classify the noised images')
    parser.add_argument('-a', '--analyzer', type=str, help='analyzer that compute the heatmaps')
    parser.add_argument('-b', '--batch_size', type=int, help='batch size to use during each epoch', default=50)
    parser.add_argument('-z', '--baseline', type=str, help='baseline type to compute IG', default='zero')
    parser.add_argument('-n', '--num_noise', type=int, help='number of noise baselines used by IG', default=1)
    parser.add_argument('-s', '--noise_scale', type=float, help='magnitude of the noise scale used to noise the image', default=5e-1)
    parser.add_argument('-p', '--noise_percent', action='store_true', help='use proportion of image as noise scale')
    parser.add_argument('-an', '--adaptive_noise', action='store_true', help='use adaptive noise')
    parser.add_argument('-af', '--adaptive_function', type=str, help='objective function used', default='aupc')
    parser.add_argument('-r', '--num_roots', type=int, help='number of noised images used', default=150)
    parser.add_argument('-k', '--kernel_size', type=int, help='size of the window of each perturbation', default=15)
    parser.add_argument('-pt', '--num_perturbs', type=int, help='number of random perturbations to evaluate', default=50)
    parser.add_argument('-l', '--num_regions', type=int, help='number of regions to perturbate', default=30)
    parser.add_argument('-d', '--draw_mode', type=int, help='perturb draw mode: 0 - uniform; 1 - gaussian according to image stats', default=0)
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
    if args.draw_mode > 1:
        print('Invalid draw mode:', args.draw_mode)
        exit()
    if args.adaptive_function not in ['aupc', 'autvc']:
        print('Invalid adaptive objective function:', args.adaptive_function)
        exit()
    return args


def run_perturbations_evaluation(dataset, model, model_name, batch_size, analyzer,
                                 noise_scale=None, num_roots=None,
                                 noise_percent=False, baseline=None, num_noise=None,
                                 kernel_size=15, num_regions=30,
                                 num_perturbs=50, draw_mode=0, adaptive_noise=False, adaptive_function=None, overwrite=False):
    # Read all scores and top10idxs for that model
    input_dir = os.path.join('output/', model_name)
    if not os.path.exists(input_dir):
        print('Model classification output not found for:', model_name)
        exit()
    input_path = os.path.join(input_dir, 'all_scores.npy')
    all_scores = np.load(input_path)
    input_path = os.path.join(input_dir, 'all_top10_idxs.npy') 
    all_top10_idxs = np.load(input_path)

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
    output_dir = os.path.join('perturbations/', analyzer, model_name)
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

    # Initialize perturbation scores
    dataset_size = len(dataset)
    all_perturb_scores = np.ones((dataset_size, num_regions + 1)) * np.inf

    # Go through each image
    for img_idx, img_filepath in enumerate(tqdm(dataset.img_filepaths, desc='Image')):
        # Check if perturbation scores exists
        img_filename = os.path.basename(img_filepath)
        perturbation_filename = 'perturbations_scores_{0}k_{1}p_{2}r_{3}d_{4}.npy'.format(
            kernel_size, num_perturbs, num_regions, draw_mode, img_filename
        )
        perturb_outpath = os.path.join(output_dir, perturbation_filename)
        if not overwrite and os.path.exists(perturb_outpath):  # ignore if the perturbation done
            print(img_filename, 'already has perturbation scores computed')
            all_perturb_scores[img_idx] = np.load(perturb_outpath)
            continue

        # Load the image heatmap
        heatmap_filepath = os.path.join(heatmap_dir, img_filename + '_hm.npy')
        if not os.path.exists(heatmap_filepath):
            print('Heatmaps not found for:', heatmap_filepath)
            continue
        heatmap = np.load(heatmap_filepath)

        # Save top score as the starting perturb score as the first region
        img_input = dataset[img_idx]['image']
        predicted_class = all_top10_idxs[img_idx, 0]
        all_perturb_scores[img_idx, 0] = all_scores[img_idx, predicted_class]

        mean_perturb_scores = compute_perturbations(
            img_input=img_input,
            model=model,
            batch_size=batch_size,
            explained_class=predicted_class,
            heatmap=heatmap,
            kernel_size=kernel_size,
            draw_mode=draw_mode,
            num_regions=num_regions,
            num_perturbs=num_perturbs
        )

        # Update all perturb scores and save individual scores
        all_perturb_scores[img_idx, 1:] = mean_perturb_scores
        np.save(perturb_outpath, all_perturb_scores[img_idx])
    
    # Save all perturbation scores
    out_filename = 'all_perturbations_scores_{0}k_{1}p_{2}r_{3}d.npy'.format(
        kernel_size, num_perturbs, num_regions, draw_mode
    )
    outpath = os.path.join(output_dir, out_filename)
    np.save(outpath, all_perturb_scores)


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

    # Load the pre-trained model
    model = MODELS[args.model_name](pretrained=True)
    model = model.to(DEVICE)
    model.eval()

    # Perform perturbation experiment
    run_perturbations_evaluation(
        dataset=dataset,
        model=model,
        model_name=args.model_name,
        batch_size=args.batch_size,
        analyzer=args.analyzer,
        noise_scale=args.noise_scale,
        num_roots=args.num_roots,
        baseline=args.baseline,
        num_noise=args.num_noise,
        kernel_size=args.kernel_size,
        num_regions=args.num_regions,
        num_perturbs=args.num_perturbs,
        draw_mode=args.draw_mode,
        adaptive_noise=args.adaptive_noise,
        adaptive_function=args.adaptive_function,
        overwrite=args.overwrite,
        noise_percent=args.noise_percent
    )

    end_time = datetime.now()
    elapsed_seconds = int((end_time - start_time).total_seconds())
    print('Start time:', start_time.strftime('%d %b %Y %H:%M:%S'))
    print('End time:', end_time.strftime('%d %b %Y %H:%M:%S'))
    print('Elapsed time:', timedelta(seconds=elapsed_seconds))