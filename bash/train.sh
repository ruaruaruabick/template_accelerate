accelerate launch --config_file configs/acc_config/default.yaml train.py \
    --config configs/default.yaml \
    --expname 'default' \
    --restore_step -1
