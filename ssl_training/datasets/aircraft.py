import os

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms
from torch.utils.data import Dataset
# import torchvision.transforms.InterpolationMode as InterpolationMode
import torchfile
import random
from PIL import Image
import numpy as np

im_dir = '/work/osaha_umass_edu/oid-aircraft-beta-1/data/images/aeroplane'
# seg_dir = '/work/osaha_umass_edu/oid-aircraft-beta-1/fh_masks'

# seg_dir = '/work/osaha_umass_edu/oid-aircraft-beta-1/data_oid/'
seg_dir = '/work/osaha_umass_edu/CUB_full/oid_trained/'
# seg_dir = '/work/osaha_umass_edu/oid-aircraft-beta-1/detcon_clus/'

# seg_dir = '/work/osaha_umass_edu/CUB_full/dino_masks/1/'


resnet_transform = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

class OIDDatasetClus(Dataset):
    def __init__(
            self,
            img_size=(224, 224),
    ):  
        with open("/work/osaha_umass_edu/oid-aircraft-beta-1/train_oid_coarse.txt", 'r') as f:
            x = f.readlines()
        # train_dat = torchfile.load('/home/osaha_umass_edu/gitcodes/CoarseSup/Posterior/anno/train.dat')
        self.img_path_list = []
        self.label_path_list = []
        for name in x:
            self.img_path_list.append(os.path.join(im_dir, name[:-5]+'.jpg'))
            self.label_path_list.append(os.path.join(seg_dir, name[:-5]+'.png'))
        self.img_size = img_size

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        im_path = self.img_path_list[index]
        lbl_path = self.label_path_list[index]
        im = Image.open(im_path).convert('RGB')
        try:
            lbl = np.load(lbl_path)
        except:
            lbl = np.array(Image.open(lbl_path))
        if len(lbl.shape) == 3:
            lbl = lbl[:, :, 0]
        # import pdb;pdb.set_trace()
        lbl = Image.fromarray(lbl.astype('uint8'))#//15)
        im1, m1 = self.transform(im, lbl, 1.0, 0.0)
        im2, m2 = self.transform(im, lbl, 0.1, 0.2)

        return im1, im2, m1, m2

    def transform(self, img, mask, g_p, s_p):
        im_shape = (img.size[1], img.size[0])
        # img = torchvision.transforms.CenterCrop(im_shape)(img)
        mask = torchvision.transforms.Resize(im_shape, interpolation=Image.NEAREST)(mask)
        
        hflip = random.random() < 0.5
        if hflip:
          img = img.transpose(Image.FLIP_LEFT_RIGHT)
          mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        crop = torchvision.transforms.RandomResizedCrop((224,224))
        params = crop.get_params(img, scale=(0.08, 1.0), ratio=(0.75, 1.33))
        img_crop = torchvision.transforms.functional.crop(img, *params)
        mask_crop = torchvision.transforms.functional.crop(mask, *params)
        img = torchvision.transforms.Resize((224,224), interpolation=Image.BICUBIC)(img_crop)
        mask = torchvision.transforms.Resize((224,224), interpolation=Image.NEAREST)(mask_crop)

        # img = torchvision.transforms.RandomResizedCrop(self.img_size, scale=(0.08, 1.0),
        #                  ratio=(0.75, 1.3333333333333333), interpolation=Image.BICUBIC)(img)
        # mask = torchvision.transforms.RandomResizedCrop(self.img_size, scale=(0.08, 1.0),
        #                  ratio=(0.75, 1.3333333333333333), interpolation=Image.NEAREST)(mask)


        jitter = random.random() < 0.8
        if jitter:
            img = torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)(img)

        gauss = random.random() < g_p
        if gauss:
            img = torchvision.transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))(img)

        img = torchvision.transforms.RandomSolarize(threshold=0.5, p=s_p)(img)

        mask = torch.from_numpy(np.array(mask)).float()
        img = torchvision.transforms.ToTensor()(img)
        img = resnet_transform(img)
        return img, mask