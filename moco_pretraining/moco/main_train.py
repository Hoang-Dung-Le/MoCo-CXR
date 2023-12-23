#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import os
import random
import shutil
import time
import warnings
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from tqdm import tqdm

import training_tools.evaluator as eval_tools
from training_tools.meters import AverageMeter
from training_tools.meters import ProgressMeter

from utils.datasets import build_dataset_chest_xray

import aihc_utils.storage_util as storage_util
import aihc_utils.image_transform as image_transform
from sklearn.metrics import roc_auc_score

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
# JBY: Decrease number of workers
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 12)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=30., type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--pretrained', default='', type=str,
                    help='path to moco pretrained checkpoint')


# Stanford AIHC modification
parser.add_argument('--exp-name', dest='exp_name', type=str, default='exp',
                    help='Experiment name')

parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

parser.add_argument("--train_list", default=None, type=str, help="file for train list")
parser.add_argument("--val_list", default=None, type=str, help="file for val list")
parser.add_argument("--test_list", default=None, type=str, help="file for test list")
parser.add_argument("--build_timm_transform", action='store_true', default=False)
parser.add_argument("--aug_strategy", default='default', type=str, help="strategy for data augmentation")
parser.add_argument("--dataset", default='chestxray', type=str)
parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
parser.add_argument('--src', action='store_true')
parser.add_argument("--norm_stats", default=None, type=str)
parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
parser.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "pixel")')
parser.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),

parser.add_argument('--save-epoch', dest='save_epoch', type=int, default=1,
                    help='Number of epochs per checkpoint save')
parser.add_argument('--from-imagenet', dest='from_imagenet', action='store_true',
                    help='use pre-trained ImageNet model')
parser.add_argument('--best-metric', dest='best_metric', type=str, default='acc@1',
                    help='metric to use for best model')
parser.add_argument('--semi-supervised', dest='semi_supervised', action='store_true',
                    help='allow the entire model to fine-tune')

parser.add_argument('--binary', dest='binary', action='store_true', help='change network to binary classif')

parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--cos-rate', default=4, type=float, metavar='CR',
                    help='Scaling factor for cos, higher the slower the decay')

parser.add_argument('--img-size', dest='img_size', type=int, default=299,
                    help='image size (Chexpert=320)')
parser.add_argument('--crop', dest='crop', type=int, default=299,
                    help='image crop (Chexpert=320)')
parser.add_argument('--maintain-ratio', dest='maintain_ratio', action='store_true',
                    help='whether to maintain aspect ratio or scale the image')
parser.add_argument('--rotate', dest='rotate', action='store_true',
                    help='to rotate image')
parser.add_argument('--optimizer', dest='optimizer', default='adam',
                    help='optimizer to use, chexpert=adam, moco=sgd')
parser.add_argument('--aug-setting', default='chexpert',
                    choices=['moco_v1', 'moco_v2', 'chexpert'],
                    help='version of data augmentation to use')
                    
best_metrics = ({'auc' : {'func': 'computeAUROC', 'format': ':6.2f', 'args': []}})
best_metric_val = 0


def computeAUROC(dataGT, dataPRED, classCount=14):
    outAUROC = []
    # print(dataGT.shape, dataPRED.shape)
    # print("ok toi da chay")
    for i in range(classCount):
        try:
            outAUROC.append(roc_auc_score(dataGT[:, i], dataPRED[:, i]))
        except:
            outAUROC.append(0.)
    return outAUROC

def evaluate(val_loader, model, computeAUROC):
    model.eval()

    gt = []
    preds = []

    if torch.cuda.is_available():
        model = model.to("cuda")  # Đưa model lên GPU nếu có

    with torch.no_grad():
        with tqdm(total=len(val_loader)) as pbar:
            for inputs, labels in val_loader:
                if torch.cuda.is_available():
                    inputs = inputs.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)

                outputs = model(inputs)
                preds.append(outputs.cpu().detach().numpy())  # Chuyển output về CPU
                gt.append(labels.cpu().numpy())

                pbar.update()

    gt = np.concatenate(gt, axis=0)
    preds = np.concatenate(preds, axis=0)

    auroc = computeAUROC(gt, preds)

    auc_each_class_array = np.array(auroc)
    result = np.average(auc_each_class_array[auc_each_class_array != 0])

    return result

def evaluate_on_train(output, targets):

    gt = []
    preds = []
    targets = targets.cpu().numpy()
    
    gt.append(targets)
    preds.append(output)
        
    gt = np.concatenate(gt, axis=0)
    preds = np.concatenate(preds, axis=0)

    auroc = computeAUROC(gt, preds)

    auc_each_class_array = np.array(auroc)
    result = np.average(auc_each_class_array[auc_each_class_array != 0])

    return result

def main():

    args = parser.parse_args()
    print(args)
    # checkpoint_folder = storage_util.get_storage_folder(args.exp_name, f'moco_lincls')
    checkpoint_folder = '/content/drive/MyDrive/dataa/LUAN_VAN/checkpoint_moco_multilabels/'
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        

    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](pretrained=args.from_imagenet)

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False
            
    num_classes = 14

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                    # remove prefix
                    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            # assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))


    criterion = nn.BCEWithLogitsLoss().cuda(args.gpu)


    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))

    assert len(parameters) == 2  # fc.weight, fc.bias
    
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(parameters, args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(parameters, args.lr,
                                     betas=(0.9, 0.999),
                                     weight_decay=args.weight_decay)


    # if args.optimizer == 'sgd':
    #     optimizer = torch.optim.SGD(model.fc.parameters(), args.lr,
    #                                 momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    # elif args.optimizer == 'adam':
    #     optimizer = torch.optim.Adam(model.fc.parameters(), args.lr,
    #                              betas=(0.9, 0.999),
    #                              weight_decay=args.weight_decay)

        

    if args.aug_setting == 'moco_v2':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        
    train_loader = build_dataset_chest_xray(split='train', args=args)
    val_loader = build_dataset_chest_xray(split='val', args=args)
    test_loader = build_dataset_chest_xray(split='test', args=args)


    train_sampler = None

    

    train_loader = torch.utils.data.DataLoader(
        train_loader, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_loader,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_loader,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    print(args.semi_supervised)
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        print(f'==> Training, epoch {epoch}')

        if torch.cuda.is_available():
            model = model.to("cuda")  # Đưa model lên GPU nếu có

        model.train()  # Đặt model ở trạng thái train

        with tqdm(total=len(train_loader)) as pbar:
            for i, (images, target) in enumerate(train_loader):
                if torch.cuda.is_available():
                    images = images.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)

                # Tính toán output
                output = model(images)

                # Tính toán loss
                loss = criterion(output, target)

                # Cập nhật trọng số
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update()

        # Đánh giá model (model vẫn ở trạng thái train)
        print("-----------evaluate-------------")

        mRocAUC = evaluate(val_loader, model, computeAUROC)
        print("auc: ", mRocAUC)

        if epoch == args.start_epoch and args.pretrained:
            sanity_check(model.state_dict(), args.pretrained, args.semi_supervised)


def save_checkpoint(checkpoint_folder, state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(checkpoint_folder, filename))
    if is_best:
        shutil.copyfile(os.path.join(checkpoint_folder, filename),
                        os.path.join(checkpoint_folder, 'model_best.pth.tar'))


def sanity_check(state_dict, pretrained_weights, semi_supervised):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    if semi_supervised:
        print('SKIPPING SANITY CHECK for semi-supervised learning')
        return

    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'fc.weight' in k or 'fc.bias' in k:
            continue

        # name in pretrained model
        k_pre = 'module.encoder_q.' + k[len('module.'):] \
            if k.startswith('module.') else 'module.encoder_q.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


# JBY: Ported over support for Cosine learning rate
def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        # TODO, JBY, is /4 an appropriate scale?
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs / args.cos_rate))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()