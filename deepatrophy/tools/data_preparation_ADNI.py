from __future__ import print_function, division
import os
import numpy as np
import random
import glob
import csv
from pathlib import Path
from datetime import datetime
from itertools import permutations

# prepare a csv file from the ADNI dataset, for DeepAtrophy training

groups = ["train", "eval", "test"]
stages = ["0", "1", "3", "5"]
root_dir = "/data/mengjin/Longi_T1_2GO_QC"
data_dir = "final_paper"
date_format = "%Y-%m-%d"
csv_dir = "/data/mengjin/DeepAtrophyPackage/DeepAtrophy/data"

# create a csv file for each group, each line is an image pair

# remove any existing csv files
csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))

for file in csv_files:
    try:
        os.remove(file)
        print(f"Deleted: {file}")
    except Exception as e:
        print(f"Error deleting {file}: {e}")

# first create a csv file for each group, each line is a single image pair for model test.

for group in groups:
    csv_list = csv_dir + "/csv_list_" + group + "_pair.csv"
    print("file saved to:", csv_list)

    with open(csv_list, 'w') as filename:
        wr = csv.writer(filename, lineterminator='\n')
        wr.writerow(["bl_fname1", "fu_fname1", "seg_fname1", "bl_time1", "fu_time1", "stage",
                 "date_diff1", "label_date_diff1", "subjectID", "side"])
        for stage in stages:
            print(group, stage)
            subject_list = root_dir + "/" + data_dir + "/subject_list_" + group + stage + ".csv"
            if os.path.exists(subject_list):
                with open(subject_list) as f:
                    for subjectID in f:
                        subjectID = subjectID.strip('\n')
                        for side in ["left", "right"]:
                            # All filenames are useful. Randomly take two baseline images and expand them into two pairs.
                            scan_list = glob.glob(root_dir + "/T1_Input_3d" + "/*/" + stage + "/" + subjectID + "*blmptrim_" + side + "_to_hw.nii.gz")
                            
                            for bl_item1 in list(scan_list):

                                fu_item1 = Path(
                                    bl_item1.replace('blmptrim_', "fumptrim_om_", 1).replace('_to_hw', 'to_hw', 1))
                                seg_item1 = Path(
                                    bl_item1.replace('blmptrim_', "blmptrim_seg_", 1).replace('_to_hw', "_tohw", 1))
                                if fu_item1.exists() and seg_item1.exists():
                                    fname1 = bl_item1.split("/")[-1]
                                    bl_time1 = datetime.strptime(fname1.split("_")[3], date_format)
                                    fu_time1 = datetime.strptime(fname1.split("_")[4], date_format)
                                    date_diff1 = (fu_time1 - bl_time1).days
                                    label_date_diff1 = float(np.greater(date_diff1, 0))
                                else:
                                    continue
                                wr.writerow(
                                    [bl_item1, fu_item1, seg_item1, bl_time1, fu_time1, stage,
                                        date_diff1,
                                        label_date_diff1, subjectID,
                                        side])

# then create a csv file for each group, each line is a double-image pair for RISI training

for group in groups:
    csv_list = csv_dir + "/csv_list_" + group + "_double_pair.csv"
    print("file saved to:", csv_list)

    with open(csv_list, 'w') as filename:
        wr = csv.writer(filename, lineterminator='\n')
        wr.writerow(["bl_fname1", "fu_fname1", "seg_fname1", "bl_fname2", "fu_fname2", "seg_fname2",
                        "bl_time1", "fu_time1", "bl_time2", "fu_time2", "stage",
                        "date_diff1", "date_diff2", "label_date_diff1", "label_date_diff2", "label_time_interval", "subjectID", "side"])
        
        for stage in stages:
            print(group, stage)
            subject_list = root_dir + "/" + data_dir + "/subject_list_" + group + stage + ".csv"
            if os.path.exists(subject_list):
                with open(subject_list) as f:
                    for subjectID in f:
                        subjectID = subjectID.strip('\n')
                        for side in ["left", "right"]:
                            # All filenames are useful. Randomly take two baseline images and expand them into two pairs.
                            scan_list = glob.glob(root_dir + "/T1_Input_3d" + "/*/" + stage + "/" + subjectID + "*blmptrim_" + side + "_to_hw.nii.gz")
                            perm = permutations(range(0, len(scan_list)), 2)
                            for bl_item1, bl_item2 in list(perm):
                                bl_item1 = scan_list[bl_item1]
                                bl_item2 = scan_list[bl_item2]

                                fu_item1 = Path(bl_item1.replace('blmptrim_', "fumptrim_om_", 1).replace('_to_hw', 'to_hw', 1))
                                mask_item1 = Path(bl_item1.replace('blmptrim_', "blmptrim_seg_", 1).replace('_to_hw', "_tohw", 1))
                                if fu_item1.exists() and mask_item1.exists():
                                    fname1 = bl_item1.split("/")[-1]
                                    bl_time1 = datetime.strptime(fname1.split("_")[3], date_format)
                                    fu_time1 = datetime.strptime(fname1.split("_")[4], date_format)
                                    date_diff1 = (fu_time1 - bl_time1).days
                                    label_date_diff1 = float(np.greater(date_diff1, 0))
                                else:    continue

                                fu_item2 = Path(bl_item2.replace('blmptrim_', "fumptrim_om_", 1).replace('_to_hw', 'to_hw', 1))
                                mask_item2 = Path(
                                    bl_item2.replace('blmptrim_', "blmptrim_seg_", 1).replace('_to_hw', "_tohw", 1))

                                if fu_item2.exists() and mask_item2.exists():
                                    fname2 = bl_item2.split("/")[-1]
                                    bl_time2 = datetime.strptime(fname2.split("_")[3], date_format)
                                    fu_time2 = datetime.strptime(fname2.split("_")[4], date_format)
                                    date_diff2 = (fu_time2 - bl_time2).days
                                    label_date_diff2 = float(np.greater(date_diff2, 0))
                                else:    continue

                                date_diff_ratio = abs(date_diff1 / date_diff2)

                                if date_diff_ratio < 0.5:
                                    label_time_interval = 0
                                elif date_diff_ratio < 1:
                                    label_time_interval = 1
                                elif date_diff_ratio < 2:
                                    label_time_interval = 2
                                else:
                                    label_time_interval = 3

                                if abs(date_diff1) > abs(date_diff2):
                                    if date_diff1 > 0:
                                        if bl_time2 <= fu_time1 and bl_time2 >= bl_time1 and fu_time2 <= fu_time1 and fu_time2 >= bl_time1:
                                            wr.writerow(
                                                [bl_item1, fu_item1, mask_item1, bl_item2, fu_item2, mask_item2, 
                                                # [bl_item1, bl_item2,
                                                 bl_time1, fu_time1, bl_time2, fu_time2, stage, date_diff1, date_diff2,
                                                 label_date_diff1, label_date_diff2, label_time_interval, subjectID, side])
                                    else:
                                        if bl_time2 <= bl_time1 and bl_time2 >= fu_time1 and fu_time2 <= bl_time1 and fu_time2 >= fu_time1:
                                            wr.writerow(
                                                [bl_item1, fu_item1, mask_item1, bl_item2, fu_item2, mask_item2, 
                                                # [bl_item1, bl_item2,
                                                 bl_time1, fu_time1, bl_time2, fu_time2, stage, date_diff1, date_diff2,
                                                 label_date_diff1, label_date_diff2, label_time_interval, subjectID, side])

                                elif abs(date_diff1) < abs(date_diff2):
                                    if date_diff2 > 0:
                                        if bl_time1 <= fu_time2 and bl_time1 >= bl_time2 \
                                                and fu_time1 <= fu_time2 and fu_time1 >= bl_time2:
                                            wr.writerow(
                                                [bl_item1, fu_item1, mask_item1, bl_item2, fu_item2, mask_item2, 
                                                # [bl_item1, bl_item2,
                                                bl_time1, fu_time1, bl_time2, fu_time2, stage,
                                                date_diff1, date_diff2,
                                                label_date_diff1, label_date_diff2, label_time_interval, subjectID,
                                                side])
                                    else:
                                        if bl_time1 <= bl_time2 and bl_time1 >= fu_time2 \
                                                and fu_time1 <= bl_time2 and fu_time1 >= fu_time2:
                                            wr.writerow(
                                                [bl_item1, fu_item1, mask_item1, bl_item2, fu_item2, mask_item2, 
                                                # [bl_item1, bl_item2,
                                                bl_time1, fu_time1, bl_time2, fu_time2, stage,
                                                date_diff1, date_diff2,
                                                label_date_diff1, label_date_diff2, label_time_interval, subjectID,
                                                side])



