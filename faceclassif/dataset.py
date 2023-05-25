# Define custom datasets to load the faces for training
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms.functional as F
import pandas as pd
import os
import matplotlib.pyplot as plt
from math import ceil, floor
import numpy as np


class Faceset(Dataset):
    def __init__(self, img_dir_path, labels_path):
        self.img_dir_path = img_dir_path
        self.labels_path = labels_path
        labels = pd.read_csv(self.labels_path, sep = '\t', header = None, names = ['image','label', 'genre'])
        labels = labels.astype({"label": int, "genre": int})
        labels['label'] = labels['label'].map(lambda x: 0 if x == -1 else x)
        self.labels = labels
        
    def __getitem__(self, index):
        item_info = self.labels.iloc[index] 
        item_name = item_info['image']
        path_img = os.path.join(self.img_dir_path, item_name)
        img = read_image(path_img)
        return img, item_info['label'], item_name

    def __len__(self):
        return len(self.labels)
    
    def plot(self, indices):
        # max 5 columns
        ncols = 5
        fig, axs = plt.subplots(nrows=ceil(len(indices)/ncols), ncols=5, squeeze=False)
        for i in indices:
            img, _, name = self.__getitem__(i)
            img = img.detach()
            img = F.to_pil_image(img)
            axs[floor(i/ncols), i%ncols].imshow(np.asarray(img))
        plt.savefig(os.path.join("fig", f'last_image_{name}')) # name of the last image indice