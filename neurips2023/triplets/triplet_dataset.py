import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.models import alexnet
from torchvision.models.alexnet import AlexNet_Weights

from torchvision import transforms
from torchvision.datasets.folder import default_loader
from pdb import set_trace

import warnings

from torchvision import transforms

__all__ = ['get_standard_transforms', 'TripletDataset', 'show_composite_view']

current_directory = os.path.dirname(__file__)
triplet_filename = os.path.join(current_directory, 'triplets.csv')

def get_standard_transforms(img_size=256, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    inv_transform = transforms.Compose([
        transforms.Normalize(
            mean= [-m/s for m, s in zip(mean, std)],
            std= [1/s for s in std]
        ),
        transforms.ToPILImage(),
    ])

    return transform, inv_transform

class TripletDataset(object):
    def __init__(self, root_dir, csv_file=triplet_filename, transform=None, loader=default_loader,
                 drop_repeats=True):

        self.df = self.load_dataframe(csv_file, drop_repeats)
        self.root_dir = root_dir
        self.loader = loader
        self.transform = transform

    def load_dataframe(self, csv_file, drop_repeats):
        df = pd.read_csv(csv_file)
        if drop_repeats:
            df['sorted_filenames'] = df.apply(lambda row: tuple(sorted([row['file1'], row['file2']])), axis=1)
            df = df.drop_duplicates(subset='sorted_filenames').drop(columns='sorted_filenames')
        else:
            warnings.warn("The data contains duplicate filename pairs. If you aren't treating imageA different than imageB when rendering, consider setting drop_repeats=True. For example, if you are presenting a 50/50 blend, you should drop_repeats. But, if you are rendering side-by-side, the order of the pair matters, and the A-B piar is different than B-A, so you can include both (drop_repeats=False)")

        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        file1 = os.path.join(self.root_dir, row.file1)
        file2 = os.path.join(self.root_dir, row.file2)
        file3 = os.path.join(self.root_dir, row.file3)

        img1 = self.loader(file1).convert('RGB')
        img2 = self.loader(file2).convert('RGB')
        img3 = self.loader(file3).convert('RGB')

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        return [img1,img2,img3],[row.label1,row.label2,row.label3]
    
def show_composite_view(imgs, labels, inv_transform, composite_func):
    # Check that there are at least two images to create a composite
    if len(imgs) < 2:
        raise ValueError("Need at least two images to create a composite")

    # Create a composite image and add it to the beginning of the imgs and labels lists
    composite = inv_transform(composite_func(imgs[0], imgs[1]))
    imgs = [inv_transform(img) for img in imgs] + [composite]  # transform all images
    labels = labels + ["composite"]

    # Determine the number of subplots needed - based on the number of images
    num_imgs = len(imgs)
    cols = 4  # You can choose how many columns you want
    rows = num_imgs // cols + (1 if num_imgs % cols else 0)

    # Create subplots
    fig, axs = plt.subplots(rows, cols, figsize=(15, 5 * rows))  # adjust the size as needed
    axs = axs.ravel()  # Flatten the array of axs if more than one row exists

    # Plot each image in its subplot
    for i in range(len(axs)):
        if i < num_imgs:
            axs[i].imshow(imgs[i])  # This assumes the image is in a format that imshow can interpret
            axs[i].set_title(labels[i])
            axs[i].axis('off')  # Hide axes
        else:
            # Hide empty subplots
            axs[i].axis('off')

    plt.tight_layout()
    plt.show()    