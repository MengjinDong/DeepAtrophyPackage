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
from nilearn.image import resample_img
import matplotlib.pyplot as plt

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg') # MUST BE CALLED BEFORE IMPORTING plt, or qt5agg

plt.ion()   # interactive mode


class LongitudinalDataset3D(Dataset):
    """ AD longitudinal dataset."""

    def __init__(self, csv_list, augment=None,
                 max_angle = 0, 
                 rotate_prob = 0.5, 
                 downsample_factor = 1,
                 output_size = [1, 1, 1]):
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
        self.downsample_factor = downsample_factor

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
        # side = self.image_frame.iloc[idx]["side"]
        side = self.image_frame.iloc[idx]["side"] if "side" in self.image_frame.columns else ""


        random_bl1 = ''.join(random_bl1)
        random_fu1 = self.image_frame.iloc[idx]["fu_fname1"]
        random_mask1 = str(self.image_frame.iloc[idx]["seg_fname1"])

        random_bl2 = ''.join(random_bl2)
        random_fu2 = self.image_frame.iloc[idx]["fu_fname2"]
        random_mask2 = str(self.image_frame.iloc[idx]["seg_fname2"])

        ########### load images
        if self.downsample_factor == 1:
            bl_cube1 = nib.load(random_bl1).get_fdata().squeeze()
            fu_cube1 = nib.load(random_fu1).get_fdata().squeeze()
            if os.path.exists(random_mask1):
                mask_cube1 = nib.load(random_mask1).get_fdata().squeeze()
            else:
                mask_cube1 = (bl_cube1 > 1).astype(float)

            bl_cube2 = nib.load(random_bl2).get_fdata().squeeze()
            fu_cube2 = nib.load(random_fu2).get_fdata().squeeze()
            if os.path.exists(random_mask2):
                mask_cube2 = nib.load(random_mask2).get_fdata().squeeze()
            else:
                mask_cube2 = (bl_cube2 > 1).astype(float)
        else:
            ########### downsample image after loading
            bl_cube1_nii = nib.load(random_bl1)
            downsampled_bl_cube1 = resample_img(bl_cube1_nii, target_affine = np.eye(3)*self.downsample_factor, interpolation='continuous')
            bl_cube1 = downsampled_bl_cube1.get_fdata().squeeze()

            fu_cube1_nii = nib.load(random_fu1)
            downsampled_fu_cube1 = resample_img(fu_cube1_nii, target_affine = np.eye(3)*self.downsample_factor, interpolation='continuous')
            fu_cube1 = downsampled_fu_cube1.get_fdata().squeeze()

            if os.path.exists(random_mask1):
                mask_cube1_nii = nib.load(random_mask1)
                downsampled_mask_cube1 = resample_img(mask_cube1_nii, target_affine = np.eye(3)*self.downsample_factor, interpolation='continuous')
                mask_cube1 = downsampled_mask_cube1.get_fdata().squeeze()
            else:
                mask_cube1 = (bl_cube1 > 1).astype(float)

            bl_cube2_nii = nib.load(random_bl2)
            downsampled_bl_cube2 = resample_img(bl_cube2_nii, target_affine = np.eye(3)*self.downsample_factor, interpolation='continuous')
            bl_cube2 = downsampled_bl_cube2.get_fdata().squeeze()

            fu_cube2_nii = nib.load(random_fu2)
            downsampled_fu_cube2 = resample_img(fu_cube2_nii, target_affine = np.eye(3)*self.downsample_factor, interpolation='continuous')
            fu_cube2 = downsampled_fu_cube2.get_fdata().squeeze()

            if os.path.exists(random_mask2):
                mask_cube2_nii = nib.load(random_mask2)
                downsampled_mask_cube2 = resample_img(mask_cube2_nii, target_affine = np.eye(3)*self.downsample_factor, interpolation='continuous')
                mask_cube2 = downsampled_mask_cube2.get_fdata().squeeze()
            else:
                mask_cube2 = (bl_cube2 > 1).astype(float)

            ########### end downsample image after loading

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

        # # Create a subplot with 2x2 layout
        # fig, axes = plt.subplots(2, 4, figsize=(10, 6))

        # image_3d = [bl_cube1, fu_cube1, bl_cube1 - fu_cube1, mask_cube1, \
        #             bl_cube2, fu_cube2, bl_cube2 - fu_cube2, mask_cube2]
        
        # title = ["BL1", "FU1", "BL1-FU1", "mask1", "BL2", "FU2", "BL2-FU2", "mask2"]

        # # Plot each slice in grayscale
        # for i, ax in enumerate(axes.flat):
        #     ax.imshow(image_3d[i][24, :, :], cmap="gray")  # Show the slice
        #     ax.set_title(f"{title[i]}")
        #     ax.axis("off")  # Hide axes

        # # Adjust layout and display
        # # plt.tight_layout()
        # plt.show()
        # plt.savefig(f"/home/mengjin/Documents/ADNI_Whole_brain/test_output{idx}.png")
        # print(f"Saved plot to test_output{idx}.png")


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
                 max_angle=0, 
                 rotate_prob=0.5, 
                 downsample_factor = 1,
                 output_size=[1, 1, 1]):
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
        self.downsample_factor = downsample_factor

        self.image_frame = pd.read_csv(csv_list)

        # with open(csv_list, 'r') as f:
        #     reader = csv.reader(f)
        #     self.image_frame = list(reader)

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
        # side = self.image_frame.iloc[idx]["side"]
        side = self.image_frame.iloc[idx]["side"] if "side" in self.image_frame.columns else ""


        random_bl1 = ''.join(random_bl1)
        random_fu1 = self.image_frame.iloc[idx]["fu_fname1"]
        random_mask1 = str(self.image_frame.iloc[idx]["seg_fname1"])

        # load images
        if self.downsample_factor == 1:
            bl_cube1 = nib.load(random_bl1).get_fdata().squeeze()
            fu_cube1 = nib.load(random_fu1).get_fdata().squeeze()
            if os.path.exists(random_mask1):
                mask_cube1 = nib.load(random_mask1).get_fdata().squeeze()
            else:
                mask_cube1 = (bl_cube1 > 1).astype(float)

        else:
            ########### downsample image after loading
            bl_cube1_nii = nib.load(random_bl1)
            downsampled_bl_cube1 = resample_img(bl_cube1_nii, target_affine = np.eye(3)*self.downsample_factor, interpolation='continuous')
            bl_cube1 = downsampled_bl_cube1.get_fdata().squeeze()

            fu_cube1_nii = nib.load(random_fu1)
            downsampled_fu_cube1 = resample_img(fu_cube1_nii, target_affine = np.eye(3)*self.downsample_factor, interpolation='continuous')
            fu_cube1 = downsampled_fu_cube1.get_fdata().squeeze()

            if os.path.exists(random_mask1):
                mask_cube1_nii = nib.load(random_mask1)
                downsampled_mask_cube1 = resample_img(mask_cube1_nii, target_affine = np.eye(3)*self.downsample_factor, interpolation='continuous')
                mask_cube1 = downsampled_mask_cube1.get_fdata().squeeze()
            else:
                mask_cube1 = (bl_cube1 > 1).astype(float)

            ########### end downsample image after loading

        # # Create a subplot with 2x2 layout
        # fig, axes = plt.subplots(2, 2, figsize=(10, 6))

        # image_3d = [bl_cube1, fu_cube1, bl_cube1 - fu_cube1, mask_cube1]
        
        # title = ["BL1", "FU1", "BL1-FU1", "mask1"]

        # # Plot each slice in grayscale
        # for i, ax in enumerate(axes.flat):
        #     slice_2d = image_3d[i][60, :, :]

        #     print(f"slice_2d shape: {slice_2d.shape}, i = {title[i]}")
        #     print(f"slice_2d min: {np.min(slice_2d)}, i = {title[i]}")
        #     print(f"slice_2d max: {np.max(slice_2d)}, i = {title[i]}")
        #     print(f"slice_2d mean: {np.mean(slice_2d)}, i = {title[i]}")
        #     ax.imshow(slice_2d, cmap="gray", vmin=np.min(slice_2d), vmax=np.max(slice_2d))
        #     ax.set_title(f"{title[i]}")
        #     ax.axis("off")  # Hide axes

        # # Adjust layout and display
        # # plt.tight_layout()
        # plt.show()
        # plt.savefig(f"/home/mengjin/Documents/ADNI_Whole_brain/test_output{idx}.png")
        # print(f"Saved plot to test_output{idx}.png")

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
        mask_cube1 = image_list1[2]

        # # Create a subplot with 2x2 layout
        # fig, axes = plt.subplots(2, 2, figsize=(10, 6))

        # image_3d = [bl_cube1, fu_cube1, bl_cube1 - fu_cube1, mask_cube1]
        
        # title = ["BL1", "FU1", "BL1-FU1", "mask1"]

        # # Plot each slice in grayscale
        # for i, ax in enumerate(axes.flat):
        #     slice_2d = image_3d[i][:, :, 32]
        #     print(f"slice_2d shape: {slice_2d.shape}, i = {title[i]}")
        #     print(f"slice_2d min: {np.min(slice_2d)}, i = {title[i]}")
        #     print(f"slice_2d max: {np.max(slice_2d)}, i = {title[i]}")
        #     print(f"slice_2d mean: {np.mean(slice_2d)}, i = {title[i]}")
        #     ax.imshow(slice_2d, cmap="gray", vmin=np.min(slice_2d), vmax=np.max(slice_2d))
        #     ax.set_title(f"{title[i]}")
        #     ax.axis("off")  # Hide axes

        # # Adjust layout and display
        # # plt.tight_layout()
        # plt.show()
        # plt.savefig(f"/home/mengjin/Documents/ADNI_Whole_brain/test_output{idx}.png")
        # print(f"Saved plot to test_output{idx}.png")


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
