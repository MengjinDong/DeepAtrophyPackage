export PYTHONPATH=$PYTHONPATH:/data/mengjin/DeepAtrophyPackage/DeepAtrophy/deepatrophy/src

cd /data/mengjin/DeepAtrophyPackage/DeepAtrophy

DATA_DIR='/data/mengjin/DeepAtrophyPackage/DeepAtrophy/data'

# run deepatrophy training

python3 -m deepatrophy run_training --train-double-pairs $DATA_DIR/csv_list_train_double_pair.csv \
    --eval-double-pairs $DATA_DIR/csv_list_eval_double_pair.csv \
    --test-double-pairs $DATA_DIR/csv_list_test_double_pair.csv \
    --train-pairs $DATA_DIR/csv_list_train_pair.csv \
    --eval-pairs $DATA_DIR/csv_list_eval_pair.csv \
    --test-pairs $DATA_DIR/csv_list_test_pair.csv \
    --ROOT "/data/mengjin/DeepAtrophyPackage/DeepAtrophy/DeepAtrophy"

# run deepatrophy testing

python3 -m deepatrophy run_test \
    --train-pairs $DATA_DIR/csv_list_train_pair.csv \
    --eval-pairs $DATA_DIR/csv_list_eval_pair.csv \
    --test-pairs $DATA_DIR/csv_list_test_pair.csv \
    --resume-all "/data/mengjin/DeepAtrophyPackage/DeepAtrophy/DeepAtrophy/Model/xxx.pth" \
    --ROOT "/data/mengjin/DeepAtrophyPackage/DeepAtrophy/DeepAtrophy"
