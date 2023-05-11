import os
import argparse
import json
from pathlib import Path

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models
import torchvision
import torchfile
from PIL import Image
import clip

import utils
import vision_transformer as vits
from fcn import fcn_res50

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from pytorch_pretrained_vit import ViT

im_dir = '../data_dir/CUB_200_2011/images_extracted/'


class ImageLabelDataset(Dataset):
    def __init__(
            self,
            mode,
            img_size=(224, 224),
    ):
        if mode == 'train':
            datfile = torchfile.load('../ssl_training/anno/train.dat')
        else:
            datfile = torchfile.load('../ssl_training/anno/val.dat')
        with open('../data_dir/CUB_200_2011/cub_classes.json', 'r') as f:
            self.class_list = json.load(f)
        self.img_path_list = []
        for name, _ in datfile.items():
            if name.decode() == 'Black_Tern_0079_143998.jpg':
                continue
            self.img_path_list.append(name.decode())
        self.img_size = img_size
        print(len(self.img_path_list))
        self.mode = mode

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        im_path = os.path.join(im_dir, self.img_path_list[index])
        im = Image.open(im_path).convert('RGB')
        class_id = np.asarray([self.class_list[self.img_path_list[index]]-1])
        im = self.transform(im)
        return im, torch.from_numpy(class_id)

    def transform(self, img):
        im_shape = (min(int(img.size[0]*0.875), int(img.size[1]*0.875)), min(int(img.size[0]*0.875), int(img.size[1]*0.875)))
        img = torchvision.transforms.CenterCrop(im_shape)(img) 
        img = img.resize(self.img_size)     
        img = transforms.ToTensor()(img)
        return img

valid_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])
def eval_linear(args):

    cudnn.benchmark = True

    # ============ building network ... ============
    if args.arch in vits.__dict__.keys():
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    elif args.arch == "resnet50":
        model = fcn_res50(pretrained=False)
    else:
        print(f"Unknown architecture: {args.arch}")
        sys.exit(1)


    model.cuda()
    model.eval()
    # load weights to evaluate
    if args.arch in vits.__dict__.keys():
        utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    elif args.arch == "resnet50":
        state_dict = torch.load(args.pretrained_weights)['state_dict']
        for key in list(state_dict.keys()):
            state_dict['res.'+key] = state_dict.pop(key)
        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            model.load_state_dict(state_dict, strict=False)
            print(e)
    print(f"Model {args.arch} built.")


    # ============ preparing data ... ============
    if args.dataset == "birds":
        dataset_train = ImageLabelDataset(mode='train')
        dataset_val = ImageLabelDataset(mode='val')
    elif args.dataset == "aircrafts":
        dataset_train = torchvision.datasets.FGVCAircraft(root='/work/pi_smaji_umass_edu/datasets/', split = 'trainval', transform=valid_transform)
        dataset_val = torchvision.datasets.FGVCAircraft(root='/work/pi_smaji_umass_edu/datasets/', split = 'test', transform=valid_transform)
    
    train_loader = DataLoader(dataset_train, batch_size=256, shuffle=False, num_workers=8, pin_memory=False)
    val_loader = DataLoader(dataset_val, batch_size=256, shuffle=False, num_workers=8, pin_memory=False)

    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

 
    train_feat, train_labels = train(model, train_loader, args.n_last_blocks)
    val_feat, val_labels = validate_network(val_loader, model, args.n_last_blocks)
    logisticRegr = LogisticRegression(max_iter=10000)
    logisticRegr.fit(train_feat, train_labels)
    print("Score = " ,logisticRegr.score(val_feat,val_labels))
 
resnet_transform = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
        

def train(model, loader, n):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    train_feat = []
    train_labels = []
    for (inp, target) in loader:
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        input_img_tensor = []
        for b in range(inp.size(0)):
            if inp.size(1) == 4:
                input_img_tensor.append(resnet_transform(inp[b][:-1,:,:]))
            else:
                input_img_tensor.append(resnet_transform(inp[b]))

        inp = torch.stack(input_img_tensor)
        # forward
        with torch.no_grad():
            if args.arch == "resnet50":
                output = model.get_encoder_output(inp)
            else:
                output = model(inp)

        train_feat.append(output)
        train_labels.append(target.squeeze(-1))   
    train_feat = torch.cat(train_feat,dim=0).detach().cpu().numpy()
    train_labels = torch.cat(train_labels,dim=0).detach().cpu().numpy()

    return train_feat, train_labels
 

@torch.no_grad()
def validate_network(val_loader, model, n):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    val_feat = []
    val_labels = []
    for (inp, target) in val_loader:
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        input_img_tensor = []
        for b in range(inp.size(0)):
            if inp.size(1) == 4:
                input_img_tensor.append(resnet_transform(inp[b][:-1,:,:]))
            else:
                input_img_tensor.append(resnet_transform(inp[b]))

        inp = torch.stack(input_img_tensor)

        # forward
        with torch.no_grad():
            if args.arch == "resnet50":
                output = model.get_encoder_output(inp)
            else:
                output = model(inp)

        val_feat.append(output)
        val_labels.append(target.squeeze(-1))   
    val_feat = torch.cat(val_feat,dim=0).detach().cpu().numpy()
    val_labels = torch.cat(val_labels,dim=0).detach().cpu().numpy()


    return val_feat, val_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture', choices=["vit_small", "resnet50"])
    parser.add_argument('--dataset', type=str, default="birds", choices=["birds", "aircrafts"])
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    args = parser.parse_args()
    eval_linear(args)
