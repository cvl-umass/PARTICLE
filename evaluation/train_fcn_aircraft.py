"""
Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
Licensed under The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import sys
sys.path.append('../')
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
import argparse
import gc
import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import glob
from torchvision import transforms
from PIL import Image
from utils.data_util import *
import json
import pickle
import time
from fcn_dino import *
import random
import cv2
import numpy as np



def main(data_path, args, resume, max_data=0, uncertainty_portion=0):
    exp_path = args['exp_dir']

    base_path = os.path.join(exp_path, "deeplab_class_%d_checkpoint_%d_filter_out_%f" %(args['testing_data_number_class'],
                                                                                        int(max_data),
                                                                                        uncertainty_portion))
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    print("Model dir,", base_path)
    num_class = args['testing_data_number_class']


    # dump_data = []
    # all_pickle = glob.glob(data_path + '/*.pickle')

    # used_image = []
    # for p in all_pickle:
    #     with open(p, 'rb') as f:
    #         curr_dict = pickle.load(f)

    #     for dd in curr_dict:
    #         if not dd['image_name'] in used_image:
    #             used_image.append(dd['image_name'] )
    #             dump_data.append(dd)
    # if max_data > 0:
    #     dump_data = dump_data[:max_data]
    # stylegan_images = [data['image_name'] for data in dump_data]
    # stylegan_labels = [data['image_label_name'] for data in dump_data]

    # stylegan_images.sort()
    # stylegan_labels.sort()
    # if uncertainty_portion > 0:
    #     sort_by_uncertainty = sorted(dump_data, key=lambda k: k['uncertrainty_score'])
    #     filter_out_num = int(len(sort_by_uncertainty) * uncertainty_portion)
    #     sort_by_uncertainty = sort_by_uncertainty[30:-filter_out_num+ 30]
    #     out_idx = range(len(sort_by_uncertainty))
    #     stylegan_images = [sort_by_uncertainty[idx]['image_name'] for idx in out_idx]
    #     stylegan_labels = [sort_by_uncertainty[idx]['image_label_name'] for idx in out_idx]

    stylegan_images = []
    stylegan_labels = []

    img_path_base = "/work/osaha_umass_edu/oid-aircraft-beta-1/data/images/aeroplane/"
    lbl_path_base = "/work/osaha_umass_edu/oid-aircraft-beta-1/data_oid/"

    with open("/work/osaha_umass_edu/oid-aircraft-beta-1/train_oid.txt", 'r') as f:
        x = f.readlines()


    for line in x:
        img_path = os.path.join(img_path_base, line[:-5]+'.jpg')
        label_path = os.path.join(lbl_path_base, line[:-5]+'/'+line[:-5])
        stylegan_images.append(img_path)
        stylegan_labels.append(label_path)


    assert  len(stylegan_images) == len(stylegan_labels)
    print( "Train data length,", str(len(stylegan_labels)))

    train_data = ImageLabelDataset(img_path_list=stylegan_images,
                              label_path_list=stylegan_labels, mode=True,
                            img_size=(args['deeplab_res'], args['deeplab_res']))

    train_data = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=0)
    classifier = fcn_res50(pretrained=True, nparts=4)
    # classifier.load_state_dict(torch.load('/work/osaha_umass_edu/dino_oid_2x/checkpoint0060.pth')['teacher'], strict=False)

    
    # classifier = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True, progress=False,
    #                                                                  num_classes=21, aux_loss=None)

    # state_dict = torch.load('/home/osaha_umass_edu/datasetGAN_release/datasetGAN/renet50_detcon.pth')['model']
    # state_dict = torch.load('/work/osaha_umass_edu/checkpoints_oid_detcon_finetuneimgnetdetcon_0.05/detcon/291_checkpoint.pth.tar')['state_dict']
    # state_dict = torch.load('/work/osaha_umass_edu/checkpoints_oid_detconclus_finetuneimgnetdetcon_0.1/detcon/381_checkpoint.pth.tar')['state_dict']
    # state_dict = torch.load('/work/osaha_umass_edu/checkpoints_seconditer_oid_detcon_0.05/detcon/1_checkpoint.pth.tar')['state_dict']



    # for key in list(state_dict.keys()):
    #     state_dict[key.replace('module.encoder.','')] = state_dict.pop(key)

    # for key in list(state_dict.keys()):
    #     state_dict['res.'+key] = state_dict.pop(key)


    # for key in list(state_dict.keys()):
    #     if 'decoder' in key:
    #         state_dict.pop(key)

    # try:
    #     classifier.load_state_dict(state_dict, strict=True)
    # except Exception as e:
    #     classifier.load_state_dict(state_dict, strict=False)
    #     print(e)

    
    # classifier.aux_classifier[4] = nn.Conv2d(256,14,1)

    # classifier.load_state_dict(torch.load('2020_02_19_15_08_41_model_best.pth.tar')['state_dict'])

    # state_dict = torch.load('/home/osaha/ContrastSeg/ckpt_epoch_800_resnet50_inat.pth')['model']

    # for key in list(state_dict.keys()):
    #     state_dict[key.replace('encoder.module.', '')] = state_dict.pop(key)

    # classifier.load_state_dict(state_dict)

    # classifier.decoder[3] = UpBlock(64, 64, upsample=True)
    # classifier.decoder.add_module("4",nn.Conv2d(64,4,1,1))

    # if resume != "":
    #     checkpoint = torch.load(resume)
    #     classifier.load_state_dict(checkpoint['model_state_dict'])

    classifier.cuda()
    # classifier = torch.nn.DataParallel(classifier)
    classifier.train()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.0001)

    resnet_transform = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])


    for epoch in range(0,200):
        time1 = time.time()
        for i, da, in enumerate(train_data):
            if da[0].shape[0] != 8:
                continue
            if i % 10 == 0:
                gc.collect()

            classifier.train()

            optimizer.zero_grad()
            img, mask = da[0], da[1]

            img = img.cuda()
            mask = mask.cuda()

            input_img_tensor = []
            for b in range(img.size(0)):
                if img.size(1) == 4:
                    input_img_tensor.append(resnet_transform(img[b][:-1,:,:]))
                else:
                    input_img_tensor.append(resnet_transform(img[b]))

            input_img_tensor = torch.stack(input_img_tensor)

            y_pred = classifier(input_img_tensor)#['out']
            loss = criterion(y_pred, mask)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(epoch, 'epoch', 'iteration', i, 'loss', loss.item())

        if epoch>100 and (epoch+1)%5==0:

            model_path = os.path.join(base_path, 'deeplab_epoch_' + str(epoch) + '.pth')

            print('Save to:', model_path)
            torch.save({'model_state_dict': classifier.state_dict()},
                       model_path)
            time2 = time.time()
            # print('******************* ye lo time dekh lo ********************')
        # print(time2-time1)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--exp', type=str)
    parser.add_argument('--resume', type=str,  default="")

    parser.add_argument('--max_data', type=int,  default=0)
    parser.add_argument('--uncertainty_portion', type=float,  default=0)


    args = parser.parse_args()

    opts = json.load(open(args.exp, 'r'))
    print("Opt", opts)

    path =opts['exp_dir']
    if os.path.exists(path):
        pass
    else:
        os.system('mkdir -p %s' % (path))
        print('Experiment folder created at: %s' % (path))

    main(args.data_path, opts, args.resume, args.max_data, args.uncertainty_portion)


