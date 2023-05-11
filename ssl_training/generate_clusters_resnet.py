from __future__ import print_function
import os
import sys
import time
import argparse
import torch
import torch.nn as nn

from utils_resnet.utils import AverageMeter, cine
from utils_resnet.resnet import InsResNet50, InsResNet18
from datasets.cub_projector import CUBDatasetProj
from datasets.aircraft_projector import OIDDatasetProj
import numpy as np
from sklearn.cluster import KMeans
import cv2

torch.manual_seed(1996)
torch.cuda.manual_seed(1996)
np.random.seed(1996)


def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=32, help='num of workers to use')

    # model definition
    parser.add_argument('--model', type=str, default='resnet50', 
                        choices=['resnet50', 'resnet18'])
    parser.add_argument('--trained_model_path', type=str, default=None, help='pretrained ssl')
    parser.add_argument('--val_layer', type=int, default=4, help='num layer in hypercol')

    # shapes
    parser.add_argument('--image_size', type=int, default=224, help='image size') 
    parser.add_argument('--val_out_size', type=int, default=64, help='output size')

    # dataset
    parser.add_argument('--dataset', type=str, default='birds', choices=['birds', 'aircrafts'])

    # model path and name  
    parser.add_argument('--model_name', type=str, default='feature projector') 
    parser.add_argument('--model_path', type=str, default='./logs') # path to store the models

    # GPU setting
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    # log_path
    parser.add_argument('--save_dir', default='./clus', type=str, metavar='PATH', help='path to save clustering outputs')
    # use hypercolumn or single layer output
    parser.add_argument('--val_use_hypercol', action='store_true', help='use HC as representations during testing')

    opt = parser.parse_args()

    return opt



def main():
    global best_error
    best_error = np.Inf

    args = parse_option()
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    torch.manual_seed(0)

    if args.dataset=='birds':
        train_dataset = CUBDatasetProj()
    elif args.dataset=='aircrafts':
        train_dataset = OIDDatasetProj()

    print('Number of training images: %d' % len(train_dataset))
    loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False, sampler=None)

    # create model
    input_size = args.image_size
    pool_size = int(input_size / 2**5)
    args.val_output_shape = (args.val_out_size, args.val_out_size)

    if args.model == 'resnet50':
        model = InsResNet50(pool_size=pool_size)
    elif args.model == 'resnet18':
        model = InsResNet18(width=1, pool_size=pool_size)
    else:
        raise NotImplementedError('model not supported {}'.format(args.model))
    
    model = nn.DataParallel(model)

    model = model.cuda()
    forward_cluster(loader, model, args)


def forward_cluster(loader, 
                    model,  # pretrained ssl
                    opt):
    cine(opt.save_dir)
    batch_time = AverageMeter()
    data_time = AverageMeter()

    end = time.time()
    for idx, (input, im_name) in enumerate(loader):
        # measure data loading time
        model.eval()
        data_time.update(time.time() - end)

        input = input.cuda(opt.gpu, non_blocking=True)
        input = input.float()

        with torch.no_grad():
            feat = model(input, opt.val_layer, opt.val_use_hypercol, opt.val_output_shape)             
            feat = feat.detach()
            feat = feat.reshape(feat.shape[0],feat.shape[1],-1)
            feat = feat.permute(0,2,1)
            feat = feat.cpu().numpy()

            for b_i in range(feat.shape[0]):
                kmeans = KMeans(n_clusters=15, random_state=np.random.randint(0,1000), max_iter=500).fit(feat[b_i])
                img = kmeans.labels_.reshape(opt.val_out_size,opt.val_out_size)
                cv2.imwrite(os.path.join(opt.save_dir, im_name[b_i][:-4]+'.png'), img)
 
            del(feat)
            batch_time.update(time.time() - end)
            end = time.time()

            # print info
            if idx % opt.print_freq == 0:
                print(
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(idx, len(loader), batch_time=batch_time,
                    data_time=data_time))
                sys.stdout.flush()
    return

if __name__ == '__main__':
    main()
