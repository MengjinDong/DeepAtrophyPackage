#!/bin/bash
set -x -e

export PYTHONPATH=$PYTHONPATH:/data/mengjin/DeepAtrophyPackage/DeepAtrophy/deepatrophy/src

cd /data/mengjin/DeepAtrophyPackage/DeepAtrophy

# 2015-01-05,  2017-04-24,  2018-05-08,  2019-05-16
baseline_image='/data/mengjin/DeepAtrophyPackage/DeepAtrophy/trimmed_image/2015-01-05_002_S_1155_T1w_trim.nii.gz'
followup_image='/data/mengjin/DeepAtrophyPackage/DeepAtrophy/trimmed_image/2017-04-24_002_S_1155_T1w_trim.nii.gz'
workdir='/data/mengjin/DeepAtrophyPackage/DeepAtrophy/global_alignment'
prefix='002_S_1155_2015-01-05_2017-04-24_T1w'

mkdir -p $workdir

template_image='/data/mengjin/DeepAtrophyPackage/DeepAtrophy/template/template.nii.gz'
template_seg="/data/mengjin/DeepAtrophyPackage/DeepAtrophy/template/refspace_meanseg_left.nii.gz"

workdir='/data/mengjin/DeepAtrophyPackage/DeepAtrophy/bounding_box'
prefix='002_S_1155_2015-01-05_2017-04-24_T1w' 

baseline_image='/data/mengjin/DeepAtrophyPackage/DeepAtrophy/trimmed_image/2015-01-05_002_S_1155_T1w_trim.nii.gz'
# followup_image = '/data/mengjin/DeepAtrophyPackage/DeepAtrophy/trimmed_image/2017-04-24_002_S_1155_T1w_trim.nii.gz'

# global alignment and obtain mask
python3 -m deepatrophy obtain_image_pair --baseline-image $baseline_image \
        --followup-image $followup_image \
        --workdir $workdir \
        --prefix $prefix \
        --template-image $template_image \
        --template-mask $template_seg \
        --get-ROI \
        --side left
        