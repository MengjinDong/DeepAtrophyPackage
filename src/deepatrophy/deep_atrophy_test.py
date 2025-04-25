#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 15:28:29 2018

@author: mengjin
ResNet training from python example

"""
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import random
import shutil
import time
import warnings
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import argparse
from . import data_loading_longi_3d as long
from . import utility as util
from . import longi_models
from . import generate_model
# from .param import parse_opts
import csv

# import matplotlib
# matplotlib.use('qt5agg') # MUST BE CALLED BEFORE IMPORTING plt, or qt5agg

def main(args):

    model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

    print("ROOT is ", args.ROOT)

    if not os.path.exists(args.ROOT + '/Model'):
        os.makedirs(args.ROOT + '/Model')

    if not os.path.exists(args.ROOT + '/log'):
        os.makedirs(args.ROOT + '/log')
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed or len(args.gpu) > 1: # if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        ngpus_per_node = len(args.gpu)
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):

    # sample_size = [48, 80, 64] # 245760
    sample_size = [args.input_D, args.input_H, args.input_W]

    global best_prec1
    args.gpu = gpu[0]

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
        print("Current Device is ", torch.cuda.get_device_name(0))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # create model2:
    if args.pretrained:
        print("=> Model (date_diff): using pre-trained model '{}_{}'".format(args.model, args.model_depth))
        pretrained_model = models.__dict__[args.arch](pretrained=True)
    else: # input: two channels.
        print("=> Model (date_diff regression): creating model '{}_{}'".format(args.model, args.model_depth))
        pretrained_model = generate_model.generate_model(args) # good for resnet
        save_folder = "{}/Model/{}{}".format(args.ROOT, args.model, args.model_depth)

    model = longi_models.ResNet_interval(pretrained_model, args.num_date_diff_classes, args.num_reg_labels)

    criterion0 = torch.nn.CrossEntropyLoss().cuda(args.gpu) # for STO loss
    criterion1 = torch.nn.CrossEntropyLoss().cuda(args.gpu) # for RISI loss

    criterion = [criterion0, criterion1]

    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                 betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    # all models optionally resume from a checkpoint
    if args.resume_all:
        if os.path.isfile(args.resume_all):
            print("=> Model_all: loading checkpoint '{}'".format(args.resume_all))
            checkpoint = torch.load(args.resume_all, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
            start_epoch = checkpoint['epoch']
            print("=> Model_all: loaded checkpoint '{}' (epoch {})"
              .format(args.resume_all, checkpoint['epoch']))
        else:
            print("Model not found: '{}'".format(args.resume_all))
            print("=> Test script exited.")
            return
    else:
        print("Model does not exist. '{}' is none.".format(args.resume_all))
        print("=> Test script exited.")
        return
    

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
            
    print("batch-size = ", args.batch_size)
    print("epochs = ", args.epochs)
    print("range-weight (weight of range loss) = ", args.range_weight)
    cudnn.benchmark = True
    # print(model)
    
    # Data loading code
    test_augment = ['normalize', 'crop']
        
    model_pair = longi_models.ResNet_pair(model.modelA, args.num_date_diff_classes)
    torch.cuda.set_device(args.gpu)
    model_pair = model_pair.cuda(args.gpu)

    model_name = args.resume_all[:-8]

    #############################################################################
    if args.train_double_pairs:
        # load these datasets for test, so use test_augment all the time.
        print("=> Test on double pairs for Train Set")
        train_dataset = long.LongitudinalDataset3D(
                args.train_double_pairs,
                test_augment, 
                args.max_angle,
                args.rotate_prob,
                sample_size)
        
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                # sampler = train_sampler,
                num_workers=args.workers, pin_memory=True)
        
        util.validate(train_loader,
                        model,
                        criterion,
                        model_name + "_train_double_pair",
                        range_weight=args.range_weight)
        
    if args.eval_double_pairs:
        
        print("=> Test on double pairs for Validate Set")
        eval_dataset = long.LongitudinalDataset3D(
                args.eval_double_pairs,
                test_augment,
                args.max_angle,
                args.rotate_prob,
                sample_size)
        
        eval_loader = torch.utils.data.DataLoader(
                eval_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.workers, pin_memory=True)
        
        prec = util.validate(eval_loader,
                    model,
                    criterion,
                    model_name + "_eval_double_pair",
                    range_weight = args.range_weight)
        
    if args.test_double_pairs:

        print("=> Test on double pairs for Test Set")
        test_dataset = long.LongitudinalDataset3D(
                args.test_double_pairs,
                test_augment,
                args.max_angle,
                args.rotate_prob,
                sample_size)

        test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.workers, pin_memory=True)
        
        util.validate(test_loader,
                        model,
                        criterion,
                        model_name + "_test_double_pair",
                        range_weight=args.range_weight)


    ################################################################################################

    # test on only the basic sub-network (STO loss)
    model_pair = longi_models.ResNet_pair(model.modelA, args.num_date_diff_classes)
    torch.cuda.set_device(args.gpu)
    model_pair = model_pair.cuda(args.gpu)

    if args.train_pairs:
        
        print("=> Test on a single image pair for Train Set")
        train_pair_dataset = long.LongitudinalDataset3DPair(
                args.train_pairs,
                test_augment,
                args.max_angle,
                args.rotate_prob,
                sample_size)

        train_pair_loader = torch.utils.data.DataLoader(
                train_pair_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.workers, pin_memory=True)
        
        util.validate_pair(train_pair_loader,
                model_pair,
                criterion,
                model_name + "_train_pair",
                epoch=args.epochs,
                print_freq=args.print_freq)

    if args.eval_pairs:

        print("=> Test on a single image pair for Eval Set")
        eval_pair_dataset = long.LongitudinalDataset3DPair(
                args.eval_pairs,
                test_augment,
                args.max_angle,
                args.rotate_prob,
                sample_size)

        eval_pair_loader = torch.utils.data.DataLoader(
                eval_pair_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.workers, pin_memory=True)

        util.validate_pair(eval_pair_loader,
                      model_pair,
                      criterion,
                      model_name + "_eval_pair",
                      args.epochs,
                      args.print_freq)
        
        
    if args.test_pairs:

        print("=> Test on a single image pair for Test Set")
        test_pair_dataset = long.LongitudinalDataset3DPair(
                args.test_pairs,
                test_augment,
                args.max_angle,
                args.rotate_prob,
                sample_size)

        test_pair_loader = torch.utils.data.DataLoader(
                test_pair_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.workers, pin_memory=True)

        util.validate_pair(test_pair_loader,
                      model_pair,
                      criterion,
                      model_name + "_test_pair",
                      args.epochs,
                      args.print_freq)


class DeepAtrophyTestLauncher:

    def __init__(self, parse):

        parse.add_argument(
            '--model',
            default='resnet',
            type=str,
            help='(resnet | preresnet | wideresnet | resnext | densenet | '
        )

        parse.add_argument(
            '--model_depth',
            default=50,
            type=int,
            help='Depth of resnet (10 | 18 | 34 | 50 | 101)'
        )

        parse.add_argument(
            '--input-channels',
            default=2,
            type=float,
            metavar='M',
            help='input channel of network.\n' +
                'input_channels = 1: sample include 2 images, each with one bl or fu image.\n' +
                'input_channels = 2: sample include one image with 2 channels (bl and fu) \n' +
                'input_channels = 3: sample include one image with 3 channels, with the third blank.\n'
        )

        parse.add_argument(
            '--num_date_diff_classes',
            default=5, # number of output parameters of the basic sub-network
            type=int,
            help="Number of date difference classes" # for resnet output
        )

        parse.add_argument(
            '--num_reg_labels',
            default = 4,
            type=int,
            help="Number of regression channels. Default 1, since regression gives 1 output"
        )

        parse.add_argument(
            '--resume-all',
            default='',
            help='path to the latest checkpoint (default: none)'
        )

        parse.add_argument(
            '--train-pairs',
            default="",
            help='A csv file containing pairs of training data'
        )

        parse.add_argument(
            '--eval-pairs',
            default="",
            help='A csv file containing pairs of evaluation data'
        )

        parse.add_argument(
            '--test-pairs',
            default="",
            help='A csv file containing pairs of test data'
        )

        parse.add_argument(
            '--trt-pairs',
            default="",
            help='A csv file containing pairs of test-retest data'
        )

        parse.add_argument(
            '--train-double-pairs',
            default="",
            help='A csv file containing pairs of training data'
        )

        parse.add_argument(
            '--eval-double-pairs',
            default="",
            help='A csv file containing pairs of evaluation data'
        )

        parse.add_argument(
            '--test-double-pairs',
            default="",
            help='A csv file containing pairs of test data'
        )

        parse.add_argument(
            '--pretrained',
            dest='pretrained',
            action='store_true',
            # can be store_true (pretrained with no modified architecture),
            # store_false (pretrained network with modified architecture)
            help='use pre-trained model'
        )

        parse.add_argument(
            '-e', '--evaluate',
            dest='evaluate',
            action='store_true',
            help='option to evaluate before starting training'
        )

        parse.add_argument(
            '-t', '--test',
            dest='test',
            action='store_true',
            help='option to test model after training on test set (not validation set)'
        )

        parse.add_argument(
            '--range-weight',
            default=1,
            type=int,
            metavar='N',
            help='weight of range loss'
        )

        parse.add_argument(
            '--ROOT',
            metavar='DIR',
            default="/data/mengjin/DeepAtrophyPackage/DeepAtrophy/DeepAtrophy",
            help='directory to save models and logs'
        )

        parse.add_argument(
            '-b', '--batch-size',
            default=60,
            type=int,  # 300 (for convnet) or 60 (for resnet)
            metavar='N', help='mini-batch size (default: 20)'
        )

        parse.add_argument(
            '--early-stop',
            default=0,
            type=int,
            metavar='N', help='flag whether to do early stopping or not'
        )

        parse.add_argument(
            '-j', '--workers',
            default=12,
            type=int,
            metavar='N',
            help='number of data loading workers (default: 4)'
        )

        parse.add_argument(
            '--lr', '--learning-rate',
            default=0.001,
            type=float,
            metavar='LR',
            help='initial learning rate'
        )

        parse.add_argument(
            '--epochs',
            default=15,
            type=int,
            metavar='N',
            help='number of total epochs to run for the categorical regression'
        )

        parse.add_argument(
            '--momentum',
            default=0.9,
            type=float,
            metavar='M',
            help='momentum'
        )

        parse.add_argument(
            '--get-prec',
            default=1,
            type=float,
            metavar='M',
            help='flag on whether to get precision of training or eval data'
        )

        parse.add_argument(
            '--weight-decay', '--wd',
            default=1e-4,
            type=float,
            metavar='W', help='weight decay (default: 1e-4)'
        )

        parse.add_argument(
            '--print-freq', '-p',
            default=20,
            type=int,
            metavar='N',
            help='print frequency (default: 20)'
        )

        parse.add_argument(
            '--eval-freq',
            default=2,
            type=int,
            metavar='N',
            help='eval frequency (default: 5)'
        )

        parse.add_argument(
            '--world-size',
            default=1,
            type=int,
            help='number of distributed processes'
        )

        parse.add_argument(
            '--dist-url',
            default='tcp://224.66.41.62:23456',
            type=str,
            help='url used to set up distributed training'
        )

        parse.add_argument(
            '--dist-backend',
            default='gloo',
            type=str,
            help='distributed backend'
        )

        parse.add_argument(
            '--max_angle',
            default=15,
            type=int,
            metavar='N',
            help='max angle of rotation when doing data augmentation'
        )

        parse.add_argument(
            '--rotate_prob',
            default=0.5,
            type=int,
            metavar='N',
            help='probability of rotation in each axis when doing data augmentation'
        )

        parse.add_argument(
            '--seed',
            default=None,
            type=int,
            help='seed for initializing training. '
        )

        parse.add_argument(
            '--gpu',
            default=[0],
            nargs='+',
            type=int,
            help='GPU id to use.'
        )

        parse.add_argument(
            '--multiprocessing-distributed',
            action='store_true',
            help='Use multi-processing distributed training to launch '
                'N processes per node, which has N GPUs. This is the '
                'fastest way to use PyTorch for either single node or '
                'multi node data parallel training'
        )

        # [48, 80, 64] -> [24, 40, 32]
        parse.add_argument(
            '--input_D',
        default=48,
            type=int,
            help='Input size of depth'
        )

        parse.add_argument(
            '--input_H',
            default=80,
            type=int,
            help='Input size of height'
        )

        parse.add_argument(
            '--input_W',
            default=64,
            type=int,
            help='Input size of width'
        )

        parse.add_argument(
            '--patience',
            default=20,
            type=int,
            help='Early stopping patience')

        parse.add_argument(
            '--tolerance',
            default=0,
            type=float,
            help='Early stopping tolerance of accuracy')

        parse.add_argument(
            '--resnet_shortcut',
            default='B',
            type=str,
            help='Shortcut type of resnet (A | B)')

        parse.add_argument(
            '--no_cuda', action='store_true', help='If true, cuda is not used.')

        parse.add_argument(
            '--manual_seed', default=1, type=int, help='Manually set random seed'
        )

        parse.add_argument(
            '--pretrain-path', default="/data/mengjin/MedicalNet/pretrain", 
            type=str, help='Pretrained model path'
        )

        # Set the function to run
        parse.set_defaults(func = lambda args : self.run(args))

    def run(self, args):

        main(args)

# if __name__ == '__main__':

#     # Create a parser with subparsers f
#     parse = argparse.ArgumentParser(description="DeepAtrophy: a Longitudinal Package for Brain Progression Estimation (test)")
#     subparsers = parse.add_subparsers(help='sub-command help')

#     # Set up the parser for sampling from a collection of CRASHS directories
#     deepatrophy_train = subparsers.add_parser('run_test', help='run deep atrophy test')
#     deepatrophy_train.set_defaults(func=main)

#     args = parse.parse_args()
#     args.func(args)
