import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
from torch.utils.data import DataLoader

from attribution import integrated_gradients
from attribution.dataset import ImageNetValDataset
from attribution.constants import *


def parse_args():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Generate integrated gradients')
    parser.add_argument('-m', '--model_name', type=str, help='name of the model used to classify')
    parser.add_argument('-b', '--batch_size', type=int, help='batch size to use during each epoch', default=50)
    parser.add_argument('-k', '--steps', type=int, help='number of steps along path', default=50)
    parser.add_argument('-z', '--baseline_type', type=str, help='baseline type to use', default='zero')
    parser.add_argument('-n', '--num_noise', type=int, help='number of noise baselines to use', default=1)
    parser.add_argument('-i', '--num_image', type=int, help='number of image data to use from the first', default=1000)
    parser.add_argument('-o', '--overwrite', action='store_true', help='overwrite the output')
    args = parser.parse_args()
    if args.model_name not in MODELS:
        print('Invalid model name:', args.model_name)
        exit()
    if args.baseline_type not in IG_BASELINES:
        print('Invalid IG baseline type:', args.baseline_type)
        exit()
    return args


def run_ig_experiment(dataset, model, model_name, batch_size, baseline_type,
                      num_noise, steps, overwrite=False):
    # Read all scores and top10idxs for that model
    input_dir = os.path.join('output/', model_name)
    if not os.path.exists(input_dir):
        print('Model classification output not found for:', model_name)
        exit()
    input_path = os.path.join(input_dir, 'all_top10_idxs.npy')
    all_top10_idxs = np.load(input_path)
    input_path = os.path.join(input_dir, 'all_scores.npy')
    all_scores = np.load(input_path)

    # Check output directory
    out_dir = os.path.join('heatmaps/ig/', model_name, baseline_type)
    if baseline_type == 'noise':
        out_dir = os.path.join(out_dir, str(num_noise) + 'N')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Generate IG and heatmap for each image
    for img_idx, img_filepath in enumerate(tqdm(dataset.img_filepaths, desc='Image')):
        # Initialize output filepath
        img_filename = os.path.basename(img_filepath)
        outpath = os.path.join(out_dir, img_filename + '_hm.npy')
        if not overwrite and os.path.exists(outpath):  # ignore if the heatmap is already generated
            print(img_filepath, 'already has heatmap generated')
            continue

        # Retrieve the image data and predicted class
        predicted_class = all_top10_idxs[img_idx, 0]
        img_input = dataset[img_idx]['image']
        input_score = all_scores[img_idx, predicted_class]
        # print('input score:', input_score)
        
        if baseline_type == 'zero':
            baseline = torch.zeros_like(img_input)
            baseline = NORMALIZE_TRANSFORM(baseline)
            ig = integrated_gradients(
                inputs=img_input,
                model=model,
                batch_size=batch_size,
                baseline=baseline,
                explained_class=predicted_class,
                steps=steps
            )
            # To check for completeness:
            # out = model(torch.unsqueeze(baseline, 0))
            # print('baseline score:',  out[:, explained_class].detach().numpy()[0])
            # print('IG sum:', np.sum(ig))
        elif baseline_type == 'noise':
            all_igs = []
            for _ in range(num_noise):
                baseline = torch.rand_like(img_input)  # uniform between 0 and 1
                baseline = NORMALIZE_TRANSFORM(baseline)
                ig = integrated_gradients(
                    inputs=img_input,
                    model=model,
                    batch_size=batch_size,
                    baseline=baseline,
                    explained_class=predicted_class,
                    steps=steps
                )
                all_igs.append(ig)
            ig = np.mean(np.array(all_igs), axis=0)

        # Save IG heatmap
        heatmap = np.sum(ig, axis=0)  # aggregate along color channel
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

    # Perform IG experiment
    run_ig_experiment(
        dataset=dataset,
        model=model,
        model_name=args.model_name,
        batch_size=args.batch_size,
        baseline_type=args.baseline_type,
        num_noise=args.num_noise,
        steps=args.steps,
        overwrite=args.overwrite
    )

    end_time = datetime.now()
    elapsed_seconds = int((end_time - start_time).total_seconds())
    print('Start time:', start_time.strftime('%d %b %Y %H:%M:%S'))
    print('End time:', end_time.strftime('%d %b %Y %H:%M:%S'))
    print('Elapsed time:', timedelta(seconds=elapsed_seconds))
