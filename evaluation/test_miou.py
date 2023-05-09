"""
Edited over code from https://github.com/nv-tlabs/datasetGAN_release/
"""
import sys
sys.path.append('../')
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
import argparse
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import glob
from utils.data_util import *
import json
from fcn_dino import fcn_dino
from fcn import fcn_res50
from datasets import BirdsImageLabelDataset, AircraftsImageLabelDataset
import cv2
import numpy as np

def cross_validate(args):
    cp_path = args.ckpt_dir
    base_path = os.path.join(cp_path, "cross_validation")
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    num_class = args.classes

    cps_all = glob.glob(cp_path + "/*")

    cp_list = [data for data in cps_all if '.pth' in data and 'best' not in data]
    cp_list.sort()

    images = []
    labels = []

    if args.dataset == "birds":
        img_path_base = '../data_dir/PASCUB_Birds/CUB_Parts/images_test'
        lbl_path_base = '../data_dir/PASCUB_Birds/CUB_Parts/parts_test'


        for file in os.listdir(img_path_base):
            img_path = os.path.join(img_path_base, file)
            label_path = os.path.join(lbl_path_base, file[:-3]+'png')
            images.append(img_path)
            labels.append(label_path)
    
    elif args.dataset == "aircrafts":
        img_path_base = "../data_dir/oid-aircraft-beta-1/data/images/aeroplane/"
        lbl_path_base = "../data_dir/oid-aircraft-beta-1/data_oid/"

        with open("../ssl_training/anno/val_oid.txt", 'r') as f:
            x = f.readlines()

        for line in x:
            img_path = os.path.join(img_path_base, line[:-5]+'.jpg')
            label_path = os.path.join(lbl_path_base, line[:-5]+'/'+line[:-5])
            images.append(img_path)
            labels.append(label_path)

    ids = range(num_class)

    fold_num =int( len(images) / 5)
    resnet_transform = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    if args.arch == "dino":
        classifier = fcn_dino(pretrained=False, nparts=num_class)
    elif args.arch == "res50":
        classifier = fcn_res50(pretrained=False, nparts=num_class)

    cross_mIOU = []
    cross_mIOU_classes = []

    for i in range(5):
        val_image = images[fold_num * i: fold_num *i + fold_num]
        val_label = labels[fold_num * i: fold_num *i + fold_num]
        test_image = [img for img in images if img not in val_image]
        test_label =[label for label in labels if label not in val_label]
        print("Val Data length,", str(len(val_image)))
        print("Testing Data length,", str(len(test_image)))

        if args.dataset == "birds":
            val_data = BirdsImageLabelDataset(img_path_list=val_image,
                                        label_path_list=val_label, mode=False,
                                        img_size=(224, 224))
            val_data = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)

            test_data = BirdsImageLabelDataset(img_path_list=test_image,
                                    label_path_list=test_label, mode=False,
                                    img_size=(224, 224))
            test_data = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)

        elif args.dataset == "aircrafts":
            val_data = AircraftsImageLabelDataset(img_path_list=val_image,
                                        label_path_list=val_label, mode=False,
                                        img_size=(224, 224))
            val_data = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)

            test_data = AircraftsImageLabelDataset(img_path_list=test_image,
                                    label_path_list=test_label, mode=False,
                                    img_size=(224, 224))
            test_data = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)

        best_val_miou = 0

        for resume in cp_list:
            checkpoint = torch.load(resume)
            classifier.load_state_dict(checkpoint['model_state_dict'])

            classifier.cuda()
            classifier.eval()

            unions = {}
            intersections = {}
            for target_num in ids:
                unions[target_num] = 0
                intersections[target_num] = 0

            with torch.no_grad():
                for _, da, in enumerate(val_data):

                    img, mask = da[0], da[1]

                    if img.size(1) == 4:
                        img = img[:, :-1, :, :]

                    img = img.cuda()
                    mask = mask.cuda()
                    input_img_tensor = []
                    for b in range(img.size(0)):
                        input_img_tensor.append(resnet_transform(img[b]))
                    input_img_tensor = torch.stack(input_img_tensor)

                    y_pred = classifier(input_img_tensor)

                    y_pred = torch.log_softmax(y_pred, dim=1)
                    _, y_pred = torch.max(y_pred, dim=1)
                    y_pred = y_pred.cpu().detach().numpy()

                    mask = mask.cpu().detach().numpy()
                    bs = y_pred.shape[0]

                    curr_iou = []

                    for target_num in ids:
                        y_pred_tmp = (y_pred == target_num).astype(int)
                        mask_tmp = (mask == target_num).astype(int)

                        intersection = (y_pred_tmp & mask_tmp).sum()
                        union = (y_pred_tmp | mask_tmp).sum()

                        unions[target_num] += union
                        intersections[target_num] += intersection

                        if not union == 0:
                            curr_iou.append(intersection / union)
                mean_ious = []

                for target_num in ids:
                    mean_ious.append(intersections[target_num] / (1e-8 + unions[target_num]))
                mean_iou_val = np.array(mean_ious).mean()

                if mean_iou_val > best_val_miou:
                    best_val_miou = mean_iou_val
                    unions = {}
                    intersections = {}
                    for target_num in ids:
                        unions[target_num] = 0
                        intersections[target_num] = 0

                    with torch.no_grad():
                        for _, da, in enumerate(test_data):

                            img, mask = da[0], da[1]

                            if img.size(1) == 4:
                                img = img[:, :-1, :, :]

                            img = img.cuda()
                            mask = mask.cuda()
                            input_img_tensor = []
                            for b in range(img.size(0)):
                                input_img_tensor.append(resnet_transform(img[b]))
                            input_img_tensor = torch.stack(input_img_tensor)

                            y_pred = classifier(input_img_tensor)#['out']
                            y_pred = torch.log_softmax(y_pred, dim=1)
                            _, y_pred = torch.max(y_pred, dim=1)
                            y_pred = y_pred.cpu().detach().numpy()

                            mask = mask.cpu().detach().numpy()

                            curr_iou = []

                            for target_num in ids:
                                y_pred_tmp = (y_pred == target_num).astype(int)
                                mask_tmp = (mask == target_num).astype(int)

                                intersection = (y_pred_tmp & mask_tmp).sum()
                                union = (y_pred_tmp | mask_tmp).sum()

                                unions[target_num] += union
                                intersections[target_num] += intersection

                                if not union == 0:
                                    curr_iou.append(intersection / union)


                            img = img.cpu().numpy()
                            img =  img * 255.
                            img = np.transpose(img, (0, 2, 3, 1)).astype(np.uint8)

                        test_mean_ious = []

                        for target_num in ids:
                            test_mean_ious.append(intersections[target_num] / (1e-8 + unions[target_num]))
                        best_test_miou = np.array(test_mean_ious).mean()
                        best_array_miou = test_mean_ious

                        print("Best IOU ,", str(best_test_miou), "CP: ", resume)

        cross_mIOU.append(best_test_miou)
        cross_mIOU_classes.append(best_array_miou)

    print(cross_mIOU)
    print(" cross validation mean:" , np.mean(cross_mIOU) )
    print(" cross validation std:", np.std(cross_mIOU))
    result = {"Cross validation mean": np.mean(cross_mIOU), "Cross validation std": np.std(cross_mIOU), "Cross validation":cross_mIOU }
    with open(os.path.join(cp_path, 'cross.json'), 'w') as f:
        json.dump(result, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="birds", choices=["birds", "aircrafts"])
    parser.add_argument('--arch', type=str, default="res50", choices=["res50", "dino"])
    parser.add_argument('--ckpt_dir', type=str,  default="")

    args = parser.parse_args()

    if args.dataset == "birds":
        args.classes = 11
    if args.dataset == "aircraft":
        args.classes = 4

    cross_validate(args)