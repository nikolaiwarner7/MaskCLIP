#!/usr/bin/env bash

CONFIG=$1
CONFIG=$1
GPUS=$'auto' # use '4' or 'auto'
PORT=${PORT:-29530}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/COCO_produce_RGBSI_maps_from_colab.py $CONFIG --launcher pytorch ${@:3}
