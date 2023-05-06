import sys
sys.path.append('../')
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import cv2

class BirdsImageLabelDataset(Dataset):
    def __init__(
            self,
            img_path_list,
            label_path_list,
            mode,
            img_size=(128, 128),
    ):
        self.img_path_list = img_path_list
        self.label_path_list = label_path_list
        self.img_size = img_size
        self.mode = mode

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        im_path = self.img_path_list[index]
        lbl_path = self.label_path_list[index]
        im = np.array(Image.open(im_path))
        try:
            lbl = np.load(lbl_path)
        except:
            lbl = np.array(Image.open(lbl_path))
        if len(lbl.shape) == 3:
            lbl = lbl[:, :, 0]

        xmin, xmax, ymin, ymax = np.where(lbl!=0)[0].min(), np.where(lbl!=0)[0].max(), np.where(lbl!=0)[1].min(), np.where(lbl!=0)[1].max()
        im_new = im[max(0,xmin-20):min(xmax+20,im.shape[0]),max(0,ymin-20):min(ymax+20, im.shape[1])]
        lbl = lbl[max(0,xmin-20):min(xmax+20,im.shape[0]),max(0,ymin-20):min(ymax+20, im.shape[1])]
        im = Image.fromarray(im_new)

        if len(np.unique(lbl))==1:
            print(im_path)
        lbl = Image.fromarray(lbl.astype('uint8'))
        im, lbl = self.transform(im, lbl)



        return im, lbl, im_path

    def transform(self, img, lbl):
        jitter = random.random() < 0.5
        if jitter and self.mode==True:
            img = torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0)(img)

        hflip = random.random() < 0.5
        if hflip and self.mode==True:
          img = img.transpose(Image.FLIP_LEFT_RIGHT)
          lbl = lbl.transpose(Image.FLIP_LEFT_RIGHT)
          lbl = np.array(lbl)
          lbl_new = lbl.copy()
          lbl_new[np.where(lbl==4)] = 5
          lbl_new[np.where(lbl==6)] = 7
          lbl_new[np.where(lbl==8)] = 9
          lbl_new[np.where(lbl==5)] = 4
          lbl_new[np.where(lbl==7)] = 6
          lbl_new[np.where(lbl==9)] = 8
          lbl = Image.fromarray(lbl_new)


        img = img.resize((self.img_size[0], self.img_size[1]))
        lbl = lbl.resize((self.img_size[0], self.img_size[1]), resample=Image.NEAREST)
        lbl = torch.from_numpy(np.array(lbl)).long()
        img = transforms.ToTensor()(img)
        return img, lbl


class AircraftsImageLabelDataset(Dataset):
    def __init__(
            self,
            img_path_list,
            label_path_list,
            mode,
            img_size=(128, 128),
    ):
        self.img_path_list = img_path_list
        self.label_path_list = label_path_list
        self.img_size = img_size
        self.mode = mode

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        im_path = self.img_path_list[index]
        lbl_path = self.label_path_list[index]

        im = np.array(Image.open(im_path))[0:-21,:,:]
        im = Image.fromarray(im)

        nose = np.array(Image.open(lbl_path+'_nose.png'))[0:-21,:]
        VS = np.array(Image.open(lbl_path+'_VS.png'))[0:-21,:]
        wheel = np.array(Image.open(lbl_path+'_wheel.png'))[0:-21,:]
        wing = np.array(Image.open(lbl_path+'_wing.png'))[0:-21,:]

        mask = np.array(Image.open(lbl_path+'_FM.png'))[0:-21,:]

        lbl = np.stack((nose,VS,wheel,wing), axis=-1)
        
        mask = Image.fromarray(mask.astype('uint8'))
        lbl = Image.fromarray(lbl.astype('uint8'))
        im, lbl, mask = self.transform(im, lbl, mask)

        return im, lbl.permute(2,0,1), mask.permute(2,0,1)

    def transform(self, img, lbl, mask):
        jitter = random.random() < 0.5
        if jitter and self.mode==True:
            img = torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0)(img)

        hflip = random.random() < 0.5
        if hflip and self.mode==True:
          img = img.transpose(Image.FLIP_LEFT_RIGHT)
          lbl = lbl.transpose(Image.FLIP_LEFT_RIGHT)
          mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        lbl = np.array(lbl)
        mask = np.expand_dims(np.array(mask),-1)
        img = img.resize((self.img_size[0], self.img_size[1]))
        lbl = cv2.resize(lbl,(self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask,(self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_NEAREST)
        mask = np.expand_dims(np.array(mask),-1)
        mask = torch.from_numpy(np.array(mask)).long()
        lbl = torch.from_numpy(np.array(lbl)).float()
        img = transforms.ToTensor()(img)
        return img, lbl, mask