LOG_LEVEL="INFO" accelerate launch --config_file configs/acc_config/default.yaml train.py \
    --config configs/default.yaml \
    --project '' \
    --expname 'default' \
    --restore_step -1 \
    --resumeid ""
