import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
from torch.utils.data import DataLoader

from attribution import get_gradients
from attribution.dataset import ImageNetValDataset
from attribution.constants import *


def parse_args():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Generate gradients and smooth grad heatmaps')
    parser.add_argument('-m', '--model_name', type=str, help='name of the model used to classify')
    parser.add_argument('-b', '--batch_size', type=int, help='batch size to use during each epoch', default=50)
    parser.add_argument('-i', '--num_image', type=int, help='number of image data to use from the first', default=1000)
    parser.add_argument('-s', '--smooth', action='store_true', help='use smooth grad')
    parser.add_argument('-p', '--noise_scale', type=float, help='percentage noise scale', default=15)
    parser.add_argument('-n', '--num_noise', type=int, help='number of noise inputs to use', default=50)
    parser.add_argument('-o', '--overwrite', action='store_true', help='overwrite the output')
    args = parser.parse_args()
    if args.model_name not in MODELS:
        print('Invalid model name:', args.model_name)
        exit()
    return args


def run_grad_experiment(dataset, model, model_name, batch_size, smooth,
                        num_noise, noise_scale, overwrite=False):
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
    if smooth:
        noise_scale_folder = str(noise_scale) + '%'
        num_noise_folder = str(num_noise) + 'N'
        noise_scale = noise_scale / 100.
        out_dir = os.path.join('heatmaps/smooth-grad/', model_name)
        out_dir = os.path.join(out_dir, noise_scale_folder, num_noise_folder)
    else:
        out_dir = os.path.join('heatmaps/grad/', model_name)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)


    # Generate grad and heatmap for each image
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
        
        if smooth:
             # Generate noised dataset based on the original inputs with additive noise
            noised_inputs = torch.stack([torch.zeros_like(img_input) for _ in range(num_noise)])
            img_noise_scale = noise_scale * (np.max(img_input.numpy()) - np.min(img_input.numpy()))
            for i in range(num_noise):
                noised_inputs[i] = img_input + img_noise_scale * torch.randn_like(img_input)
            noised_dataset = torch.utils.data.dataset.TensorDataset(noised_inputs)
            noised_data_loader = DataLoader(noised_dataset, batch_size=batch_size, shuffle=False)

             # Compute the gradients w.r.t. explained class for each noised input
            gradients = get_gradients(noised_data_loader, model, explained_class=predicted_class)
            grad = np.mean(gradients, axis=0)
        else:
            single_dataset = torch.utils.data.dataset.TensorDataset(torch.unsqueeze(img_input, 0))
            single_data_loader = DataLoader(single_dataset, batch_size=1, shuffle=False)
            grad = get_gradients(
                data_loader=single_data_loader,
                model=model,
                explained_class=predicted_class
            )[0]

        # Save grad heatmap
        heatmap = np.sum(grad, axis=0)  # aggregate along color channel
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

    # Perform grad experiment
    run_grad_experiment(
        dataset=dataset,
        model=model,
        model_name=args.model_name,
        batch_size=args.batch_size,
        smooth=args.smooth,
        num_noise=args.num_noise,
        noise_scale=args.noise_scale,
        overwrite=args.overwrite
    )

    end_time = datetime.now()
    elapsed_seconds = int((end_time - start_time).total_seconds())
    print('Start time:', start_time.strftime('%d %b %Y %H:%M:%S'))
    print('End time:', end_time.strftime('%d %b %Y %H:%M:%S'))
    print('Elapsed time:', timedelta(seconds=elapsed_seconds))
