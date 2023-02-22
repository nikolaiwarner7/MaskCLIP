#!/usr/bin/env bash

CONFIG=$1
CONFIG=$1
GPUS=$'auto' # use '4' or 'auto'
PORT=${PORT:-29530}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train_RGBS+I_maskclip_model_COCO.py $CONFIG --launcher pytorch ${@:3}
