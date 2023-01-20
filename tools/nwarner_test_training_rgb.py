
import os
os.chdir('/root/MaskCLIP')
import sys
sys.path.append('/root/MaskCLIP/')
sys.path.append('/root/MaskCLIP/mmseg')
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor
import argparse
from mmcv.utils import Config, DictAction, get_git_hash
from mmcv.cnn.utils import revert_sync_batchnorm
import mmcv
#from mmseg.utils import get_device

import os.path as osp\

parser = argparse.ArgumentParser(description='Train a segmentor')

parser.add_argument('--config', default='configs/deeplabv3plus/deeplabv3plus_r50-d8_512x512_40k_voc12aug.py', help='test config file path')
 
args = parser.parse_args()
cfg = Config.fromfile(args.config)

# Get rid of aug
cfg.data.train['ann_dir'].pop(1)
cfg.data.train['split'].pop(1)
cfg.data['samples_per_gpu'] = 2

# Build the dataset
datasets = [build_dataset(cfg.data.train)]

# Build the detector
model = build_segmentor(cfg.model)
# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES


# SyncBN is not support for DP
model = revert_sync_batchnorm(model)

# Create work_dir
# Set up working dir to save files and logs.
cfg.work_dir = './work_dirs/tutorial'
cfg.gpu_ids = range(1)
cfg.seed = 5050
#cfg.device = get_device()

mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_segmentor(model, datasets, cfg, distributed=False, validate=True, 
                meta=dict())

