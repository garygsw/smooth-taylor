import os
import PIL
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset

from .constants import *


class ImageNetValDataset(Dataset):
    def __init__(self, root_dir, label_dir, synset_filepath, max_num=None, transform=None):
        """
        Args:
            root_dir (string): Directory with all the raw images
            label_dir (string): Directory with the ground truth label (in XML files)
            synset_filepath (string): Filepath of synset file
            max_num (int, optional): Optional maximum number of image to load
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.synset_filepath = synset_filepath
        if transform:
            self.transform = transform
        else:
            self.transform = DEFAULT_TRANSFORM

        self.parse_synset()
        self.img_filepaths = []
        self.labels = []

        # Prepare the images and labels
        for root, dirs, files in os.walk(self.root_dir):
            for i, filename in enumerate(sorted(files)):
                if max_num and i >= (max_num):
                    print('Read data read limit reached:', max_num)
                    break
                img_filepath = os.path.join(root, filename)
                self.img_filepaths.append(img_filepath)
                self.labels.append(self.parse_label(filename))

    def parse_synset(self):
        """ Prepare the following mappings:
            idx2synset:  dict mapping of idx to synsets
            synset2idx:  dict mapping of synset to idx
            synset2class:  dict mapping of synset to class
        """
        with open(self.synset_filepath) as f:
            synsets = [(line.split()[0], ' '.join(line.split()[1:])) for line in f.readlines()]
            self.idx2synset = {i: x[0] for i, x in enumerate(synsets)}
            self.synset2idx = {x[0]: i for i, x in enumerate(synsets)}
            self.synset2class = {x[0]: x[1] for x in synsets}

    def parse_label(self, img_filename):
        """Read the XML annotation file and retrieve the label index."""
        img_identifier, ext = os.path.splitext(img_filename)
        label_filepath = os.path.join(self.label_dir, img_identifier) + '.xml'
        tree = ET.parse(label_filepath)
        root = tree.getroot()

        label_set = set()
        for obj in root.findall('object'):
            for name in obj.findall('name'):
                idx = self.synset2idx[name.text]
                label_set.add(idx)
        if len(label_set) != 1:
            print('ERR: More than 1 label is found for ', img_filename)
            exit()
        return idx

    def __len__(self):
        return len(self.img_filepaths)

    def __getitem__(self, idx):
        """
        Returns:
            sample: dict that contains image data, label and image filepath of the indexed image
        """
        img_filepath = self.img_filepaths[idx]
        image = PIL.Image.open(img_filepath).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        if image.size()[1] == 1:  # if only 1 channel after transformation, repeat channel 3 times
            image = image.repeat([1, 3, 1, 1])

        sample = {'image': image, 'label': label, 'filepath': img_filepath}
        return sample
