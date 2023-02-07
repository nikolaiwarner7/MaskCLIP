#!/usr/bin/env bash

CONFIG=$1
GPUS=$'auto'
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train_RGBS_maskclip_model.py $CONFIG --launcher pytorch ${@:3}
