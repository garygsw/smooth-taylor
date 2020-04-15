import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
from torch.utils.data import DataLoader

from attribution import smooth_taylor
from attribution.dataset import ImageNetValDataset
from attribution.constants import *


def parse_args():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Generate smooth taylor attributions')
    parser.add_argument('-m', '--model_name', type=str, help='name of the model used to classify')
    parser.add_argument('-b', '--batch_size', type=int, help='batch size to use during each epoch', default=50)
    parser.add_argument('-s', '--noise_scale', type=float, help='magnitude of the noise scale to noise the image', default=5e-1)
    parser.add_argument('-r', '--num_roots', type=int, help='number of noised images to use', default=150)
    parser.add_argument('-i', '--num_image', type=int, help='number of image data to use from the first', default=1000)
    parser.add_argument('-p', '--noise_percent', action='store_true', help='use proportion of image as noise scale')
    parser.add_argument('-o', '--overwrite', action='store_true', help='overwrite the output')
    args = parser.parse_args()
    if args.model_name not in MODELS:
        print('Invalid model name:', args.model_name)
        exit()
    return args


def run_smooth_taylor_experiment(dataset, model, model_name, batch_size, noise_scale, num_roots, overwrite=False, noise_percent=True):
    # Read all top10idxs for that model
    input_dir = os.path.join('output/', model_name)
    if not os.path.exists(input_dir):
        print('Model classification output not found for:', model_name)
        exit()
    input_path = os.path.join(input_dir, 'all_top10_idxs.npy') 
    all_top10_idxs = np.load(input_path)

    # Check output directory
    if noise_percent:
        noise_folder = str(noise_scale) + '%'
        noise_scale = noise_scale / 100.
    else:
        noise_folder = '{:.1e}'.format(noise_scale)   # convert to scientific notation
    roots_folder = str(num_roots) + 'N'
    out_dir = os.path.join('heatmaps/smooth-taylor/', model_name, noise_folder, roots_folder)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Generate SmoothTaylor heatmap for each image
    for img_idx, img_filepath in enumerate(tqdm(dataset.img_filepaths, desc='Image')):
        # Initialize output filepath
        img_filename = os.path.basename(img_filepath)
        outpath = os.path.join(out_dir, img_filename + '_hm.npy')
        if not overwrite and os.path.exists(outpath):  # ignore if already generated
            print(img_filename, 'already has heatmap generated')
            continue

        # Retrieve the image data and predicted class
        predicted_class = all_top10_idxs[img_idx, 0]
        img_input = dataset[img_idx]['image']

        # Compute SmoothTaylor heatmaps
        attributions = smooth_taylor(
            inputs=img_input,
            model=model,
            batch_size=batch_size,
            noise_scale=noise_scale,
            num_roots=num_roots,
            explained_class=predicted_class,
            percent=noise_percent
        )
        heatmap = np.sum(attributions, axis=0)   # sum across all channels
        np.save(outpath, heatmap)


if __name__=='__main__':
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

    # Perform smooth taylor experiment
    run_smooth_taylor_experiment(
        dataset=dataset,
        model=model,
        model_name=args.model_name,
        batch_size=args.batch_size,
        noise_scale=args.noise_scale,
        num_roots=args.num_roots,
        overwrite=args.overwrite,
        noise_percent=args.noise_percent
    )

    end_time = datetime.now()
    elapsed_seconds = int((end_time - start_time).total_seconds())
    print('Start time:', start_time.strftime('%d %b %Y %H:%M:%S'))
    print('End time:', end_time.strftime('%d %b %Y %H:%M:%S'))
    print('Elapsed time:', timedelta(seconds=elapsed_seconds))