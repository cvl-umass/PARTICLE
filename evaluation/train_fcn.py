import sys
sys.path.append('../')
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import argparse
import gc
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from fcn_dino import fcn_dino
from fcn import fcn_res50
from datasets import BirdsImageLabelDataset, AircraftsImageLabelDataset



def main(args):
    exp_path = args.save_path

    base_path = os.path.join(exp_path, args.dataset + "_seg_class_%d" %(args.classes))
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    print("Model dir,", base_path)
    num_class = args.classes

    images = []
    labels = []


    if args.dataset == "birds":
        img_path_base = '/home/osaha_umass_edu/PASCUB_Birds/CUB_Parts/images_train'
        lbl_path_base = '/home/osaha_umass_edu/PASCUB_Birds/CUB_Parts/parts_train'


        for file in os.listdir(img_path_base):
            img_path = os.path.join(img_path_base, file)
            label_path = os.path.join(lbl_path_base, file[:-3]+'png')
            images.append(img_path)
            labels.append(label_path)

        img_path_base = '/home/osaha_umass_edu/PASCUB_Birds/PASCAL_Parts/images_train'
        lbl_path_base = '/home/osaha_umass_edu/PASCUB_Birds/PASCAL_Parts/parts_train'

        for file in os.listdir(img_path_base):
            img_path = os.path.join(img_path_base, file)
            label_path = os.path.join(lbl_path_base, file[:-3]+'png')
            images.append(img_path)
            labels.append(label_path)
    
    elif args.dataset == "aircrafts":
        img_path_base = "/work/osaha_umass_edu/oid-aircraft-beta-1/data/images/aeroplane/"
        lbl_path_base = "/work/osaha_umass_edu/oid-aircraft-beta-1/data_oid/"

        with open("/work/osaha_umass_edu/oid-aircraft-beta-1/train_oid.txt", 'r') as f:
            x = f.readlines()

        for line in x:
            img_path = os.path.join(img_path_base, line[:-5]+'.jpg')
            label_path = os.path.join(lbl_path_base, line[:-5]+'/'+line[:-5])
            images.append(img_path)
            labels.append(label_path)


    assert len(images) == len(labels)
    print("Train data length,", str(len(labels)))

    if args.dataset == "birds":
        train_data = BirdsImageLabelDataset(img_path_list=images,
                                label_path_list=labels, mode=True,
                                img_size=(224,224))
    elif args.dataset == "aircrafts":
        train_data = AircraftsImageLabelDataset(img_path_list=images,
                                label_path_list=labels, mode=True,
                                img_size=(224,224))

    train_data = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=0)
    if args.arch == "dino":
        classifier = fcn_dino(pretrained=False, nparts=num_class)
    elif args.arch == "res50":
        classifier = fcn_res50(pretrained=False, nparts=num_class)


    if args.ckpt != "":
        checkpoint = torch.load(args.ckpt)
        if args.arch == "res50":
            state_dict = checkpoint['state_dict']
            for key in list(state_dict.keys()):
                state_dict['res.'+key] = state_dict.pop(key)
            classifier.load_state_dict(state_dict, strict=False)
        elif args.arch == "dino":
            classifier.load_state_dict(checkpoint['teacher'], strict=False)

    classifier.cuda()
    classifier.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.000001)

    resnet_transform = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    tot = 0
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

            y_pred = classifier(input_img_tensor)
            loss = criterion(y_pred, mask)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(epoch, 'epoch', 'iteration', i, 'loss', loss.item())

        if epoch>0 and (epoch+1)%5==0:

            model_path = os.path.join(base_path, 'deeplab_epoch_' + str(epoch) + '.pth')

            print('Save to:', model_path)
            torch.save({'model_state_dict': classifier.state_dict()},
                       model_path)
            time2 = time.time()
        
        time2 = time.time()
        print(time2-time1)
        tot=tot+(time2-time1)
        print(tot/(epoch+1))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="birds", choices=["birds", "aircrafts"])
    parser.add_argument('--arch', type=str, default="res50", choices=["res50", "dino"])
    parser.add_argument('--ckpt', type=str,  default="")
    parser.add_argument('--save_path', type=str,  default="./birds_seg")

    args = parser.parse_args()

    if args.dataset == "birds":
        args.classes = 11
    if args.dataset == "aircraft":
        args.classes = 4

    path = args.save_path
    if os.path.exists(path):
        pass
    else:
        os.system('mkdir -p %s' % (path))
        print('Experiment folder created at: %s' % (path))

    main(args)


