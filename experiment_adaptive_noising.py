import os
import math
import numpy as np
from tqdm import tqdm
import argparse
import torch
from torch.utils.data import DataLoader

from attribution.eval import compute_aupc, compute_autvc
from attribution.dataset import ImageNetValDataset
from attribution.constants import *


def parse_args():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Perform adaptive noising per image for smooth integrated gradients')
    parser.add_argument('-m', '--model_name', type=str, help='name of the model used to classify')
    parser.add_argument('-b', '--batch_size', type=int, help='batch size to use during each epoch', default=50)
    parser.add_argument('-r', '--num_roots', type=int, help='number of noised images used', default=150)
    parser.add_argument('-f', '--obj_function', type=str, help='objective function to optimize', default='aupc')
    parser.add_argument('-ds', '--downscale', type=float, help='factor to downscale heatmap', default=1.5)
    parser.add_argument('-wms', '--width_min_size', type=int, help='minimum width dimension for downscale', default=30)
    parser.add_argument('-hms', '--height_min_size', type=int, help='minimum height dimension for downscale', default=30)
    parser.add_argument('-lp', '--lp_norm', type=int, help='norm to use to calculate TV', default=1)
    parser.add_argument('-k', '--kernel_size', type=int, help='size of the window of each perturbation', default=15)
    parser.add_argument('-p', '--num_perturbs', type=int, help='number of random perturbations to evaluate', default=50)
    parser.add_argument('-l', '--num_regions', type=int, help='number of regions to perturbate', default=30)
    parser.add_argument('-d', '--draw_mode', type=int, help='perturb draw mode: 0 - uniform; 1 - gaussian according to image stats', default=0)
    parser.add_argument('-lr', '--learning_rate', type=float, help='learning rate for variable update', default=0.1)
    parser.add_argument('-y', '--learning_decay', type=float, help='decay rate of learning rate', default=0.9)
    parser.add_argument('-c', '--max_stop_count', type=int, help='maximum stop count to terminate search', default=3)
    parser.add_argument('-x', '--max_iteration', type=int, help='maximum iterations to search', default=20)
    parser.add_argument('-i', '--num_image', type=int, help='number of image data to use from the first', default=1000)
    parser.add_argument('-o', '--overwrite', action='store_true', help='overwrite the output')
    args = parser.parse_args()
    if args.model_name not in MODELS:
        print('Invalid model name:', args.model_name)
        exit()
    if args.obj_function not in ['aupc', 'autvc']:
        print('Invalid objective function:', args.obj_function)
        exit()
    return args


def adapt_noise(img_input, model, num_roots, learning_rate, learning_decay,
                max_stop_count, max_iteration, obj_function, params):
    # Initialize values
    if obj_function == 'aupc':
        compute_auc = compute_aupc
    elif obj_function == 'autvc':
        compute_auc = compute_autvc
    current_noise = np.mean(np.abs(img_input.numpy())) # initialize noise = absolute mean
    params['noise_scale'] = current_noise
    current_heatmap, current_score, current_auc = compute_auc(**params)
    # print('initial noise:', current_noise)
    # print('initial auc:', current_auc)
    best_noise = current_noise
    best_auc = current_auc
    best_score = current_score
    best_heatmap = current_heatmap
    stop_count = 0
    lr = learning_rate
    iteration = 1

    while iteration <= max_iteration:
        # Find direction
        params['noise_scale'] = math.fabs(current_noise + lr)
        current_heatmap, current_score, search_auc = compute_auc(**params)
        if search_auc > current_auc:
            current_noise = math.fabs(current_noise - lr)
            params['noise_scale'] = current_noise
            current_heatmap, current_score, search_auc = compute_auc(**params)
        else:
            current_noise = math.fabs(current_noise + lr)
        print('update noise -', 'noise:', current_noise)
        
        # Early stopping
        if search_auc > current_auc:  # worse
            if stop_count < max_stop_count: 
                lr = lr * learning_decay  # reduce lr
                stop_count += 1
            else:
                print('finished -', 'best auc:', best_auc, 'best noise:', best_noise)
                break  # exit the loop
        else:  # improved
            stop_count = 0
            if search_auc < best_auc:
                best_auc = search_auc
                best_noise = current_noise
                best_score = current_score
                best_heatmap = current_heatmap
                print('update best -', 'best auc:', best_auc, 'best noise:', best_noise)
        current_auc = search_auc
        iteration += 1
    return best_noise, best_heatmap, best_auc, best_score


def run_adapt_noise_experiment(dataset, model, model_name, batch_size, transform,
                               num_roots, obj_function, learning_rate, learning_decay,
                               max_stop_count, max_iteration, downscale, min_size, lp_norm,
                               kernel_size, num_regions, draw_mode, num_perturbs,
                               percent=False, overwrite=False):
    # Read all top10idxs and all scores for that model (only for aupc objective function)
    input_dir = os.path.join('output/', model_name)
    if not os.path.exists(input_dir):
        print('Model classification output not found for:', model_name)
        exit()
    input_path = os.path.join(input_dir, 'all_scores.npy')
    all_model_scores = np.load(input_path)
    input_path = os.path.join(input_dir, 'all_top10_idxs.npy') 
    all_top10_idxs = np.load(input_path)

    # Prepare heatmap output directory
    output_dir = os.path.join('adaptive/', model_name, obj_function)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    heatmap_dir = os.path.join('heatmaps/', 'smooth-taylor', model_name, 'adaptive', obj_function, str(num_roots) + 'N')
    if not os.path.isdir(heatmap_dir):
        os.makedirs(heatmap_dir)

    # Prepare output file formats
    all_scores = []
    if obj_function == 'aupc':
        params_format = ('{num_roots}N_{learning_rate}lr_{learning_decay}ld_'
                      '{max_stop_count}s_{kernel_size}k_{num_perturbs}p_'
                      '{num_regions}r_{draw_mode}d')
        single_params_format = params_format + '_{img_filename}'
        score_file_format = 'perturbations_scores_' + single_params_format + '.npy'
        all_scores_file_format = 'all_perturb_scores_' + params_format + '.npy'
        exp_params = {
            'num_roots': num_roots,
            'learning_rate': learning_rate,
            'learning_decay': learning_decay,
            'max_stop_count': max_stop_count,
            'kernel_size': kernel_size,
            'num_perturbs': num_perturbs,
            'num_regions': num_regions,
            'draw_mode': draw_mode
        }
    elif obj_function == 'autvc':
        params_format = ('{num_roots}N_{learning_rate}lr_{learning_decay}ld_'
                         '{max_stop_count}s_{downscale}ds_{min_size}ms_{lp_norm}l')
        single_params_format = params_format + '_{img_filename}'
        score_file_format = 'atv_scores_' + single_params_format + '.npy'
        all_scores_file_format = 'all_atv_scores_' + params_format +'.npy'
        exp_params = {
            'num_roots': num_roots,
            'learning_rate': learning_rate,
            'learning_decay': learning_decay,
            'max_stop_count': max_stop_count,
            'downscale': downscale,
            'min_size': min_size,
            'lp_norm': lp_norm
        }
    noise_file_format = 'noise_scores_' + single_params_format + '.npy'
    all_noise_file_format = 'all_noise_' + params_format + '.npy'

    # Prepare overall output file paths
    all_noise_scores_filename = all_noise_file_format.format(**exp_params)
    all_noise_scores_filepath = os.path.join(output_dir, all_noise_scores_filename)
    all_scores_filename = all_scores_file_format.format(**exp_params)
    all_scores_filepath = os.path.join(output_dir, all_scores_filename)

    # Initialize noise scores
    dataset_size = len(dataset)
    all_noise_scores = np.zeros((dataset_size, 2))  # noise, AUC score

    # Go through each image
    for img_idx, img_filepath in enumerate(tqdm(dataset.img_filepaths, desc='Image')):
        # Check if adaptive noise output already exists
        img_filename = os.path.basename(img_filepath)
        exp_params['img_filename'] = img_filename
        
        # Prepare noise score outpath
        noise_score_filename = noise_file_format.format(**exp_params)
        noise_score_outpath = os.path.join(output_dir, noise_score_filename)
        
        # Prepare score output
        scores_filename = score_file_format.format(**exp_params)
        scores_outpath = os.path.join(output_dir, scores_filename)

        # Prepare best heatmap output
        best_heatmap_outpath = os.path.join(heatmap_dir, img_filename + '_hm.npy')

        if not overwrite and os.path.exists(noise_score_outpath) and \
            os.path.exists(scores_outpath) and \
            os.path.exists(best_heatmap_outpath):  # ignore if already generated
            print(img_filename, 'already has noise scores generated')
            all_noise_scores[img_idx] = np.load(noise_score_outpath)
            all_scores.append(np.load(scores_outpath))
            continue

        # Initialize parameters
        img_input = dataset[img_idx]['image']
        predicted_class = all_top10_idxs[img_idx, 0]
        if obj_function == 'aupc':
            # Retrieve the image data, predicted class and score
            input_score = all_model_scores[img_idx, predicted_class]
            params = {
                'img_input': img_input,
                'model': model,
                'batch_size': batch_size,
                'transform': transform,
                'analyzer': 'smooth-taylor',
                'explained_class': predicted_class,
                'input_score': input_score,
                'num_roots': num_roots,
                'kernel_size': kernel_size,
                'draw_mode': draw_mode,
                'num_regions': num_regions,
                'num_perturbs': num_perturbs,
            }
        elif obj_function == 'autvc':
            params = {
                'img_input': img_input,
                'model': model,
                'batch_size': batch_size,
                'transform': transform,
                'analyzer': 'smooth-taylor',
                'explained_class': predicted_class,
                'num_roots': num_roots,
                'downscale': downscale,
                'min_size': min_size,
                'lp_norm': lp_norm,
            }

        # Find best noise scale
        best_noise, best_heatmap, best_auc, best_scores = adapt_noise(
            img_input=img_input,
            model=model,
            num_roots=num_roots,
            learning_rate=learning_rate,
            learning_decay=learning_decay,
            max_stop_count=max_stop_count,
            max_iteration=max_iteration,
            obj_function=obj_function,
            params=params
        )

        # Save the best heatmap
        np.save(best_heatmap_outpath, best_heatmap)

        # Save the best noise and auc scores
        all_noise_scores[img_idx, 0] = best_noise
        all_noise_scores[img_idx, 1] = best_auc
        all_scores.append(best_scores)
        scores_outpath = os.path.join(output_dir, scores_filename)
        np.save(scores_outpath, best_scores)
        np.save(noise_score_outpath, all_noise_scores[img_idx])

    # Save all scores
    np.save(all_scores_filepath, np.array(all_scores))
    np.save(all_noise_scores_filepath, all_noise_scores)


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

    # Perform experiment
    run_adapt_noise_experiment(
        dataset=dataset,
        model=model,
        model_name=args.model_name,
        batch_size=args.batch_size,
        transform=NORMALIZE_TRANSFORM,
        obj_function=args.obj_function,
        num_roots=args.num_roots, 
        learning_rate=args.learning_rate,
        learning_decay=args.learning_decay,
        max_stop_count=args.max_stop_count,
        max_iteration=args.max_iteration,
        overwrite=args.overwrite,
        # autvc params:
        downscale=args.downscale,
        min_size=(args.height_min_size, args.width_min_size),
        lp_norm=args.lp_norm,
        # aupc params:
        kernel_size=args.kernel_size,
        num_regions=args.num_regions,
        draw_mode=args.draw_mode,
        num_perturbs=args.num_perturbs
    )

    end_time = datetime.now()
    elapsed_seconds = int((end_time - start_time).total_seconds())
    print('Start time:', start_time.strftime('%d %b %Y %H:%M:%S'))
    print('End time:', end_time.strftime('%d %b %Y %H:%M:%S'))
    print('Elapsed time:', timedelta(seconds=elapsed_seconds))