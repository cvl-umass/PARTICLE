from utils_resnet.models import Network, MLP
from utils_resnet.losses import DetConBLoss
from utils_resnet.utils import *
from datasets.cub import CUBDatasetClus
from datasets.aircraft import OIDDatasetClus

from typing import Dict, Sequence, Tuple

import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
import shutil
import time
import argparse
from torch_ema import ExponentialMovingAverage
from math import cos, pi
import numpy as np
from datetime import datetime


torch.manual_seed(1996)
np.random.seed(1996)

import warnings
warnings.filterwarnings("ignore")

time_string = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

cine('logs')
Tee('logs/cmd_log_{}'.format("train"), 'w')


best_loss = 10000

parser = argparse.ArgumentParser(description='PARTICLE Training')
parser.add_argument('--dataset', type=str, default='birds', choices=['birds', 'aircrafts'])
parser.add_argument('--seg_dir', default='', type=str,
         help="""Part Proposals generated using clustering.""")
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=600, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=320, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.005, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1.5e-6, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--model_path', default='', type=str, metavar='PATH',
                    help='path to the pretrained model')
parser.add_argument('--save_path', default='', type=str, metavar='PATH',
                    help='path to save model')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--lr_decay', default='cos', type=str,
                    help='lr decay type')
parser.add_argument('--crop_size', default='224', type=int,
                    help='size of cropped image')


def main():
    global args, best_loss
    args = parser.parse_args()
    print(args)
    trainer = DetConB

    model_trainer = DetConB()
    params = list(model_trainer.network.parameters()) + list(model_trainer.predictor.parameters())
    optimizer = torch.optim.SGD([{'params' : params}],
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
 
    train_sampler = None
    cudnn.benchmark = True
    if args.dataset=='birds':
        train_data = CUBDatasetClus(args.seg_dir, img_size=(224, 224))
    elif args.dataset=='aircrafts':
        train_data = OIDDatasetClus(args.seg_dir, img_size=(224, 224))

    train_loader= DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)


    for epoch in range(args.start_epoch, args.epochs):
        loss = train(train_loader,
             model_trainer,
             optimizer,
             epoch)
    
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        if epoch % 10 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_trainer.network.module.encoder.state_dict(),
                'best_loss': best_loss,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args.save_path)

def train(train_loader,
             model_trainer,
             optimizer,
             epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()    
    model_trainer.network.train()
    model_trainer.predictor.train()

    end = time.time()
    train_loader_len = len(train_loader)

    for i, (im1, im2, m1, m2) in enumerate(train_loader):
        adjust_learning_rate(optimizer, epoch, i, train_loader_len)
        data_time.update(time.time() - end)

        im1 = torch.autograd.Variable(im1.cuda())
        im2 = torch.autograd.Variable(im2.cuda())
        m1 = torch.autograd.Variable(m1.cuda())
        m2 = torch.autograd.Variable(m2.cuda())

        loss_iter = model_trainer.training_step(im1, im2, m1, m2, optimizer)

        losses.update(loss_iter.item(), im1.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:

            print('Epoch: [{0}][{1}/{2}] '
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Loss yp {loss_yp.val:.3e} ({loss_yp.avg:.3e}) '.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss_yp=losses))
    return losses.avg

def save_checkpoint(state, is_best, save_dir, filename='*_checkpoint.pth.tar'):
    cine(save_dir)
    epoch = str(state['epoch'])
    filename = filename.replace('*', epoch)
    filename = os.path.join(save_dir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(save_dir, epoch+'_model_best.pth.tar'))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class DetConB(nn.Module):
    def __init__(
        self,
        num_classes: int = 15,
        num_samples: int = 10,
        backbone: str = "resnet50",
        pretrained: bool = False,
        downsample: int = 32,
        proj_hidden_dim: int = 4096,
        proj_dim: int = 256,
        loss_fn: nn.Module = DetConBLoss(),
    ) -> None:
        super().__init__()
        self.loss_fn = loss_fn
        self.network = Network(
            backbone=backbone,
            pretrained=pretrained,
            hidden_dim=proj_hidden_dim,
            output_dim=proj_dim,
            num_classes=num_classes,
            downsample=downsample,
            num_samples=num_samples,
        ).cuda()
        self.ema = ExponentialMovingAverage(self.network.parameters(), decay=0.996)
        self.predictor = MLP(proj_dim, proj_hidden_dim, proj_dim).cuda()
        self.network = torch.nn.DataParallel(self.network)
        self.ema = torch.nn.DataParallel(self.ema)
        self.predictor = torch.nn.DataParallel(self.predictor)


    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Sequence[torch.Tensor]:
        return self.network(x, y)

    def training_step(self, x1, x2, y1, y2, optimizer) -> torch.Tensor:
        optimizer.zero_grad()
        # encode and project

        with torch.no_grad():
            with self.ema.module.average_parameters():
                temp1_1, ema_p1, temp1_2, ema_ids1 = self(x1, y1)
                temp2_1, ema_p2, temp2_2, ema_ids2 = self(x2, y2)
        ema_p1 = ema_p1.detach()
        ema_p2 = ema_p2.detach()
        ema_ids1 = ema_ids1.detach()
        ema_ids2 = ema_ids2.detach()
        temp1_1 = temp1_1.detach()
        temp1_2 = temp1_2.detach()
        temp2_1 = temp2_1.detach()
        temp2_2 = temp2_2.detach()


        _, p1, _, ids1 = self(x1, y1)
        _, p2, _, ids2 = self(x2, y2)
        
        # predict
        q1, q2 = self.predictor(p1), self.predictor(p2)

        # compute loss
        loss = self.loss_fn(
            pred1=q1,
            pred2=q2,
            target1=ema_p1, 
            target2=ema_p2, 
            pind1=ids1,
            pind2=ids2,
            tind1=ema_ids1,
            tind2=ema_ids2,
        )
        loss.backward()
        optimizer.step()

        self.ema.to(device=next(self.network.parameters()).device)
        self.ema.module.update(self.network.parameters())

        return loss  

def adjust_learning_rate(optimizer, epoch, iteration, num_iter):
    lr0 = optimizer.param_groups[0]['lr']
    warmup_epoch = 10 
    warmup_iter = warmup_epoch * num_iter
    current_iter = iteration + epoch * num_iter
    max_iter = args.epochs * num_iter

    if args.lr_decay == 'step':
        lr = args.lr * (args.gamma ** ((current_iter - warmup_iter) / (max_iter - warmup_iter)))
    elif args.lr_decay == 'cos':
        lr0 = args.lr * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
    elif args.lr_decay == 'linear':
        lr = args.lr * (1 - (current_iter - warmup_iter) / (max_iter - warmup_iter))
    elif args.lr_decay == 'schedule':
        count = sum([1 for s in args.schedule if s <= epoch])
        lr = args.lr * pow(args.gamma, count)
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_decay))

    if epoch < warmup_epoch:
        lr = args.lr * current_iter / warmup_iter

    optimizer.param_groups[0]['lr'] = lr0

if __name__ == '__main__':
    main()