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

im_dir = '../data_dir/oid-aircraft-beta-1/data/images/aeroplane/'

resnet_transform = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

class OIDDatasetProj(Dataset):
    def __init__(
            self,
            img_size=(224, 224),
    ):  
        with open("./anno/train_oid.txt", 'r') as f:
            x = f.readlines()
        self.img_path_list = []
        for name in x:
            self.img_path_list.append(os.path.join(im_dir, name[:-5]+'.jpg'))

        self.img_path_list = sorted(self.img_path_list)
        self.img_path_list = self.img_path_list[2*(len(self.img_path_list)//3):]
        self.img_size = img_size

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        im_path = self.img_path_list[index]
        im = np.array(Image.open(im_path).convert('RGB'))[0:-21,:,:]
        im = Image.fromarray(im)

        im1 = self.transform(im)

        return im1, im_path.split('/')[-1]

    def transform(self, img):
        img = img.resize(self.img_size)  
        img = torchvision.transforms.ToTensor()(img)   
        img = resnet_transform(img)
        return img