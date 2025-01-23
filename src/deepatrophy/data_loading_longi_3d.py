#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 23:15:26 2019


Data Loading and Processing Tutorial
====================================
modified from:
 `Sasank Chilamkurthy <https://chsasank.github.io>`_


"""
from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import random
import glob
import csv
from pathlib import Path
from datetime import datetime
from . import data_aug_cpu
from itertools import permutations
import nibabel as nib
import pandas as pd

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# plt.ion()   # interactive mode


class LongitudinalDataset3D(Dataset):
    """ AD longitudinal dataset."""

    def __init__(self, csv_list, augment=None,
                 max_angle = 0, rotate_prob = 0.5, output_size = [1, 1, 1]):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            groups: a list specifying the stages included in the dataset
                    (0 = A- NC,   1 = A+ NC,   2 = A- eMCI, 3 = A+ eMCI,
                     4 = A- lMCI, 5 = A+ lMCI, 6 = A- AD,   7 = A+ AD   )
            csv_list: output training or test statistics
            augment: a list specifying augmentation methods applied to this dataset
            max_angle: the maximum angles applied in random rotation
            rotate_prob: probability of applying random rotation
            output_size: the desired output size after random cropping,
                         in this experiment it is [48, 80, 64]
        """
        
        self.csv_list = csv_list
        self.augment = augment
        self.date_format = "%Y-%m-%d"
        self.max_angle = max_angle
        self.rotate_prob = rotate_prob
        self.output_size = output_size

        # with open(csv_list, 'r') as f:
        #     reader = csv.reader(f)
        #     self.image_frame = list(reader)

        self.image_frame = pd.read_csv(csv_list)
            

        
    def __len__(self):
        return len(self.image_frame)

    def __getitem__(self, idx):

        random_bl1 = self.image_frame.iloc[idx]["bl_fname1"]
        random_bl2 = self.image_frame.iloc[idx]["bl_fname2"]
        bl_time1 = self.image_frame.iloc[idx]["bl_time1"]
        fu_time1 = self.image_frame.iloc[idx]["fu_time1"]
        bl_time2 = self.image_frame.iloc[idx]["bl_time2"]
        fu_time2 = self.image_frame.iloc[idx]["fu_time2"]
        stage = self.image_frame.iloc[idx]["stage"]

        date_diff1 = float(self.image_frame.iloc[idx]["date_diff1"])
        date_diff2 = float(self.image_frame.iloc[idx]["date_diff2"])

        label_date_diff1 = float(self.image_frame.iloc[idx]["label_date_diff1"])
        label_date_diff2 = float(self.image_frame.iloc[idx]["label_date_diff2"])

        label_time_interval = float(self.image_frame.iloc[idx]["label_time_interval"])
        subjectID = self.image_frame.iloc[idx]["subjectID"]
        side = self.image_frame.iloc[idx]["side"]

        random_bl1 = ''.join(random_bl1)
        random_fu1 = self.image_frame.iloc[idx]["fu_fname1"]
        random_mask1 = self.image_frame.iloc[idx]["seg_fname1"]

        random_bl2 = ''.join(random_bl2)
        random_fu2 = self.image_frame.iloc[idx]["fu_fname2"]
        random_mask2 = self.image_frame.iloc[idx]["seg_fname2"]

        bl_cube1 = nib.load(random_bl1).get_fdata().squeeze()
        fu_cube1 = nib.load(random_fu1).get_fdata().squeeze()
        mask_cube1 = nib.load(random_mask1).get_fdata().squeeze()

        bl_cube2 = nib.load(random_bl2).get_fdata().squeeze()
        fu_cube2 = nib.load(random_fu2).get_fdata().squeeze()
        mask_cube2 = nib.load(random_mask2).get_fdata().squeeze()

        # print("len_image_list = ", len(image_list))

        if 'normalize' in self.augment:
            [bl_cube1, fu_cube1] = data_aug_cpu.Normalize([bl_cube1, fu_cube1])
            [bl_cube2, fu_cube2] = data_aug_cpu.Normalize([bl_cube2, fu_cube2])

        image_list1 = [bl_cube1, fu_cube1, mask_cube1]
        image_list2 = [bl_cube2, fu_cube2, mask_cube2]

        # flip: left right 1/6; up down 1/6; front back 1/6; no flipping 1/6
        if 'flip' in self.augment:
            image_list1 = data_aug_cpu.randomFlip3d(image_list1)
            image_list2 = data_aug_cpu.randomFlip3d(image_list2)

        # Random 3D rotate image
        if 'rotate' in self.augment and self.max_angle > 0:
            image_list1 = data_aug_cpu.randomRotation3d(image_list1, self.max_angle, self.rotate_prob)
            image_list2 = data_aug_cpu.randomRotation3d(image_list2, self.max_angle, self.rotate_prob)

        if 'crop' in self.augment:
            image_list1 = data_aug_cpu.randomCrop3d(image_list1, self.output_size)
            image_list2 = data_aug_cpu.randomCrop3d(image_list2, self.output_size)

        bl_cube1 = image_list1[0]
        fu_cube1 = image_list1[1]
        bl_cube2 = image_list2[0]
        fu_cube2 = image_list2[1]


        bl_cube1 = torch.from_numpy(bl_cube1[np.newaxis, :, :, :].copy()).float()
        bl_cube2 = torch.from_numpy(bl_cube2[np.newaxis, :, :, :].copy()).float()
        fu_cube1 = torch.from_numpy(fu_cube1[np.newaxis, :, :, :].copy()).float()
        fu_cube2 = torch.from_numpy(fu_cube2[np.newaxis, :, :, :].copy()).float()

        sample = {}

        # wrap up prepared images for network input
        input_im1 = np.concatenate(\
            (bl_cube1, fu_cube1, bl_cube2, fu_cube2), axis=0)
        sample['image'] = input_im1

        sample["bl_fname1"] = random_bl1
        sample["bl_fname2"] = random_bl2

        sample["bl_time1"] = bl_time1
        sample["bl_time2"] = bl_time2

        sample["fu_time1"] = fu_time1
        sample["fu_time2"] = fu_time2

        sample['stage'] = stage

        sample['date_diff1'] = date_diff1
        sample['date_diff2'] = date_diff2

        sample['label_date_diff1'] = torch.from_numpy(np.array(label_date_diff1).copy()).float()
        sample['label_date_diff2'] = torch.from_numpy(np.array(label_date_diff2).copy()).float()
        sample['label_time_interval'] = torch.from_numpy(np.array(label_time_interval).copy()).float()

        sample['subjectID'] = subjectID
        sample['side'] = side
        sample['date_diff_ratio'] = torch.from_numpy(np.array(abs(date_diff1/date_diff2)).copy()).float()

        return sample


class LongitudinalDataset3DPair(Dataset):
    """ AD longitudinal dataset."""

    def __init__(self, csv_list, augment=None,
                 max_angle=0, rotate_prob=0.5, output_size=[1, 1, 1]):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            csv_in: blindeddays.txt, used to extract date difference
            csv_list: output training or test statistics
        """

        self.csv_list = csv_list
        self.augment = augment
        self.date_format = "%Y-%m-%d"
        self.max_angle = max_angle
        self.rotate_prob = rotate_prob
        self.output_size = output_size

        with open(csv_list, 'r') as f:
            reader = csv.reader(f)
            self.image_frame = list(reader)

    def __len__(self):
        return len(self.image_frame)

    def __getitem__(self, idx):

        random_bl1 = self.image_frame.iloc[idx]["bl_fname1"]
        bl_time1 = self.image_frame.iloc[idx]["bl_time1"]
        fu_time1 = self.image_frame.iloc[idx]["fu_time1"]
        stage = self.image_frame.iloc[idx]["stage"]

        date_diff1 = float(self.image_frame.iloc[idx]["date_diff1"])

        label_date_diff1 = float(self.image_frame.iloc[idx]["label_date_diff1"])

        subjectID = self.image_frame.iloc[idx]["subjectID"]
        side = self.image_frame.iloc[idx]["side"]

        random_bl1 = ''.join(random_bl1)
        random_fu1 = self.image_frame.iloc[idx]["fu_fname1"]
        random_mask1 = self.image_frame.iloc[idx]["seg_fname1"]

        bl_cube1 = nib.load(random_bl1).get_fdata().squeeze()
        fu_cube1 = nib.load(random_fu1).get_fdata().squeeze()
        mask_cube1 = nib.load(random_mask1).get_fdata().squeeze()

        if 'normalize' in self.augment:
            [bl_cube1, fu_cube1] = data_aug_cpu.Normalize([bl_cube1, fu_cube1])

        image_list1 = [bl_cube1, fu_cube1, mask_cube1]

        # flip: left right 1/6; up down 1/6; front back 1/6; no flipping 1/6
        if 'flip' in self.augment:
            image_list1 = data_aug_cpu.randomFlip3d(image_list1)

        # Random 3D rotate image
        if 'rotate' in self.augment and self.max_angle > 0:
            image_list1 = data_aug_cpu.randomRotation3d(image_list1, self.max_angle, self.rotate_prob)

        if 'crop' in self.augment:
            image_list1 = data_aug_cpu.randomCrop3d(image_list1, self.output_size)

        bl_cube1 = image_list1[0]
        fu_cube1 = image_list1[1]

        bl_cube1 = torch.from_numpy(bl_cube1[np.newaxis, :, :, :].copy()).float()
        fu_cube1 = torch.from_numpy(fu_cube1[np.newaxis, :, :, :].copy()).float()

        sample = {}

        # wrap up prepared images for network input
        input_im1 = np.concatenate( \
            (bl_cube1, fu_cube1), axis=0)

        sample['image'] = input_im1
        sample["bl_fname1"] = random_bl1
        sample["bl_time1"] = bl_time1
        sample["fu_time1"] = fu_time1
        sample['stage'] = stage
        sample['date_diff1'] = date_diff1
        sample['label_date_diff1'] = torch.from_numpy(np.array(label_date_diff1).copy()).float()
        sample['subjectID'] = subjectID
        sample['side'] = side

        return sample
