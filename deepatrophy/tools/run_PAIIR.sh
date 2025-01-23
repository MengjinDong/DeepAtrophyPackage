#!/bin/bash
set -x -e

export PYTHONPATH=$PYTHONPATH:/data/mengjin/DeepAtrophyPackage/DeepAtrophy/deepatrophy/src

cd /data/mengjin/DeepAtrophyPackage/DeepAtrophy

# 2015-01-05,  2017-04-24,  2018-05-08,  2019-05-16
workdir='/data/mengjin/DeepAtrophyPackage/DeepAtrophy/PAIIR'
prefix='resnet50_2020-07-08_12-39'

train_spreadsheet='/data/mengjin/DeepAtrophyPackage/DeepAtrophy/data/resnet50_2020-07-08_12-39train_0135_train_pair_update.csv'
test_spreadsheet='/data/mengjin/DeepAtrophyPackage/DeepAtrophy/data/resnet50_2020-07-08_12-39train_0135_test_pair_update.csv'
# test_double_pair_spreadsheet='/data/mengjin/DeepAtrophyPackage/DeepAtrophy/data/resnet50_2020-07-08_12-39train_0135_test.csv'
test_double_pair_spreadsheet='/data/mengjin/DeepAtrophyPackage/DeepAtrophy/data/resnet50_2020-07-08_12-39train_0135_test_modified.csv'


mkdir -p $workdir

# run PAIIR
# args STO-spreadsheet, RISI-spreadsheet are not used in the analysis.
python3 -m deepatrophy PAIIR --train-pair-spreadsheet $train_spreadsheet \
        --test-pair-spreadsheet $test_spreadsheet \
        --test-double-pair-spreadsheet $test_double_pair_spreadsheet \
        --workdir $workdir \
        --prefix $prefix \
        --min-date 180 \
        --max-date 400

# python3 -m deepatrophy PAIIR --train-pair-spreadsheet $train_spreadsheet \
#         --test-pair-spreadsheet $test_spreadsheet \
#         --test-double-pair-spreadsheet $test_double_pair_spreadsheet \
#         --workdir $workdir \
#         --prefix $prefix \
#         --min-date 400 \
#         --max-date 800
        