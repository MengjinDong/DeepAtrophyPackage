#!/bin/bash
set -x -e

export PYTHONPATH=$PYTHONPATH:/data/mengjin/DeepAtrophyPackage/DeepAtrophy/deepatrophy/src

cd /data/mengjin/DeepAtrophyPackage/DeepAtrophy


# input_image='/data/mengjin/DeepAtrophyPackage/DeepAtrophy/data/2015-01-05_002_S_1155_T1w.nii.gz'
# trimmed_image='/data/mengjin/DeepAtrophyPackage/DeepAtrophy/trimmed_image/2015-01-05_002_S_1155_T1w_trim.nii.gz'
workdir='/data/mengjin/DeepAtrophyPackage/DeepAtrophy/trimmed_image'

# input_image='/data/mengjin/DeepAtrophyPackage/DeepAtrophy/data/2017-04-24_002_S_1155_T1w.nii.gz'
# trimmed_image='/data/mengjin/DeepAtrophyPackage/DeepAtrophy/trimmed_image/2017-04-24_002_S_1155_T1w_trim.nii.gz'

input_image='/data/mengjin/DeepAtrophyPackage/DeepAtrophy/trimmed_image/2017-04-24_002_S_1155_T1w_trim.nii.gz'
trimmed_image='/data/mengjin/DeepAtrophyPackage/DeepAtrophy/trimmed_image/2017-04-24_002_S_1155_T1w_trim2.nii.gz'

# neck trim 
python3 -m deepatrophy neck_trim --input-image $input_image \
        --trimmed-image $trimmed_image \
        --workdir $workdir
        



