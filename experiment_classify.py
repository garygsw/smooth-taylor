import os
import numpy as np
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from attribution.dataset import ImageNetValDataset
from attribution.constants import *


def parse_args():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Classify the image dataset and saves all prediction outputs')
    parser.add_argument('-m', '--model_name', type=str, help='name of the model used to classify')
    parser.add_argument('-b', '--batch_size', type=int, help='batch size to use during each epoch', default=128)
    args = parser.parse_args()
    if args.model_name not in MODELS:
        print('Invalid model name:', args.model_name)
        exit()
    return args


def run_classify_experiment(dataset, model, model_name, batch_size, num_classes):
    """
    Args:
        dataset (Dataset): Containing input images.
        model (PyTorch model): Pre-trained Pytorch classifier.
        model_name (str): Name of the model.
        batch_size (int): Batch size to use during each epoch.
        num_classes (int): Total number of classes.
    """
    # Load the dataset into a data loader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    dataset_size = len(dataset)
    
    # Initialize all scores, top10 indexes and labels
    all_scores = np.zeros((dataset_size, num_classes))
    all_top10_idxs = np.zeros((dataset_size, 10), dtype=np.int32)
    all_labels = np.zeros((dataset_size), dtype=np.int32)
    all_positives = np.zeros((dataset_size), dtype=np.int32)

    # Create output directory
    out_dir = 'output/' + model_name
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Perform classification
    running_corrects = 0.0
    total_epochs = len(data_loader)
    for i_batch, sample_batch in enumerate(tqdm(data_loader, desc='Epoch')):
        print('Epoch #%d/%d:' % (i_batch + 1, total_epochs), 'batch_shape:', sample_batch['image'].size())
        inputs = sample_batch['image']
        labels = sample_batch['label'] 
        fpaths = sample_batch['filepath']
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.no_grad():
            out = model(inputs)
            _, idx = torch.max(out, 1)
            _, top10idxs = torch.topk(out, k=10, dim=1, largest=True, sorted=True, out=None)

            # Save outputs into .npy files for each individual image
            top10idxs = top10idxs.cpu().numpy()
            scores = out.cpu().numpy()
            for i, filepath in enumerate(fpaths):
                filename = os.path.basename(filepath)
                outpath = os.path.join(out_dir, filename + '_scores.npy' )
                np.save(outpath, scores[i,:])
                all_scores[i_batch * batch_size + i,:] = scores[i,:]

                outpath = os.path.join(out_dir, filename + '_top10idxs.npy' )
                np.save(outpath, top10idxs[i,:])
                all_top10_idxs[i_batch * batch_size + i,:] = top10idxs[i,:]

                all_labels[i_batch * batch_size + i] = labels[i]
                all_positives[i_batch * batch_size + i] = int(labels[i] == top10idxs[i,0])

        num_correct = torch.sum((idx == labels).float())
        running_corrects += num_correct
        epoch_acc = num_correct / float(batch_size)
        running_acc = running_corrects / dataset_size
        print('Epoch Accuracy: %.2f%%' % (epoch_acc.cpu().numpy() * 100))
        print('Running Accuracy: %.2f%%' % (running_acc.cpu().numpy() * 100))

    # Save all scores, all top10 indexes, all labels, and all positives
    outpath = os.path.join(out_dir, 'all_scores.npy')
    np.save(outpath, all_scores)
    outpath = os.path.join(out_dir, 'all_top10_idxs.npy')
    np.save(outpath, all_top10_idxs)  
    outpath = os.path.join(out_dir, 'all_labels.npy')
    np.save(outpath, all_labels)
    outpath = os.path.join(out_dir, 'all_positives.npy')
    np.save(outpath, all_positives)

if __name__=='__main__':
    args = parse_args()

    from datetime import datetime, timedelta
    start_time = datetime.now()

    # Load the dataset
    dataset = ImageNetValDataset(
        root_dir='data/images',
        label_dir='data/annotations',
        synset_filepath='rsc/synset_words.txt'
    )

    # Load the pre-trained model
    model = MODELS[args.model_name](pretrained=True)
    model = model.to(DEVICE)
    model.eval()

    # Read the class file
    with open('rsc/imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]
    num_classes = len(classes)

    # Perform classification
    run_classify_experiment(
        dataset=dataset,
        model=model,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_classes=num_classes
    )

    end_time = datetime.now()
    elapsed_seconds = int((end_time - start_time).total_seconds())
    print('Start time:', start_time.strftime('%d %b %Y %H:%M:%S'))
    print('End time:', end_time.strftime('%d %b %Y %H:%M:%S'))
    print('Elapsed time:', timedelta(seconds=elapsed_seconds))