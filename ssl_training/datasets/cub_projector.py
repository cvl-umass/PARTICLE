import os
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms
from torch.utils.data import Dataset
import torchfile
import random
from PIL import Image
import numpy as np

im_dir = '../data_dir/CUB_200_2011/CUB_200_2011/images_extracted/'

resnet_transform = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

class CUBDatasetProj(Dataset):
    def __init__(
            self,
            img_size=(224, 224),
    ):
        train_dat = torchfile.load('./anno/train.dat')
        self.img_path_list = []
        for name, _ in train_dat.items():
            if name.decode() == 'Black_Tern_0079_143998.jpg':
                continue
            self.img_path_list.append(os.path.join(im_dir, name.decode()))
        self.img_size = img_size

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        im_path = self.img_path_list[index]
        im = Image.open(im_path).convert('RGB')
        im1 = self.transform(im)
        return im1, im_path.split('/')[-1]

    def transform(self, img):
        img = img.resize(self.img_size)  
        img = torchvision.transforms.ToTensor()(img)   
        img = resnet_transform(img)
        return img