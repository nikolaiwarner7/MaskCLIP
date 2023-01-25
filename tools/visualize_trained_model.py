import shutil
import os
os.chdir('/root/MaskCLIP')
import sys
sys.path.append('/root/MaskCLIP/')
sys.path.append('/root/MaskCLIP/mmseg')
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor
from mmseg.apis import single_gpu_test
import argparse
import torch
import torch.nn as nn
from mmcv.utils import Config, DictAction, get_git_hash
from mmcv.cnn.utils import revert_sync_batchnorm
import mmcv
#from mmseg.utils import get_device
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
                         
import os.path as osp

parser = argparse.ArgumentParser(description='Train a segmentor')

parser.add_argument('--config', default='configs/maskclip_plus/zero_shot/maskclip_plus_r50_deeplabv2_r101-d8_512x512_20k_voc12aug_20.py',\
     help='test config file path')
parser.add_argument(
    '--load-from', default='test_outs/latest.pth', help='the checkpoint file to load weights from')
parser.add_argument(
    '--show-dir', default='output/trained_model_vis/', help='directory where painted images will be saved')
parser.add_argument(
    '--opacity',
    type=float,
    default=0.5,
    help='Opacity of painted segmentation map. In (0, 1] range.')

args = parser.parse_args()

#clear previous resultss
shutil.rmtree(args.show_dir)
os.mkdir(args.show_dir)

cfg = Config.fromfile(args.config)

# Changes to switch to our RGB-S data
cfg.data.train.img_dir = 'RGB_S_Images/train'
cfg.data.train.ann_dir = 'RGB_S_Annotations/train'
cfg.data.train.split = 'ImageSets/Segmentation/RGBS_train.txt'
cfg.data.train.type = 'PascalVOCDatasetRGBS'

# Normalization in 4D
# Set mean to 0.5, std 0.5 (0,1) range of data for saliency
#
cfg.data.train['pipeline'][6] = \
    {'type': 'Normalize', 'mean': [123.675, 116.28, 103.53, 0.5], 'std': [58.395, 57.12, 57.375, 0.5], 'to_rgb': True}

# Remove unnecessary augmentations (changes shape) for now
cfg.data.train['pipeline'].pop(5) # PhotoMetricDistortion

# Set parameters for validation data
cfg.data.val.img_dir = 'RGB_S_Images/val'
cfg.data.val.ann_dir = 'RGB_S_Annotations/val'
cfg.data.val.split = 'ImageSets/Segmentation/RGBS_val.txt'
cfg.data.val.type = 'PascalVOCDatasetRGBS'

# Set normalization in 4D for val
cfg.data.val['pipeline'][1]['transforms'][2] = \
    {'type': 'Normalize', 'mean': [123.675, 116.28, 103.53, 0.5], 'std': [58.395, 57.12, 57.375, 0.5], 'to_rgb': True}

# Set batch size to 2 to allow batch norm in ASPP decoder head to work
cfg.data['samples_per_gpu'] = 1

## Remove reduce zero label for our binary mask, class agnostic training  
# Otherwise produces bug with 0/255 rolled back labels
cfg.data.train['pipeline'][1]['reduce_zero_label'] = False
# Find where to do so for val too

# Remove some augmentations
#cfg.data.train['pipeline'][2] = {'type': 'Resize', 'img_scale': (512, 512), 'ratio_range': (1.0, 1.0)}
#cfg.data.train['pipeline'][3] = {'type': 'RandomCrop', 'crop_size': (512, 512), 'cat_max_ratio': 1.0}
#cfg.data.train['pipeline'][4] = {'type': 'RandomFlip', 'prob': 0.0}
# Change hook to tensorboard
cfg.log_config = dict(
interval=50,
hooks=[
    dict(type='TextLoggerHook'),
    dict(type='TensorboardLoggerHook')
])

## Add in loading annotatio[15, 16, 17, 18, 19]
cfg.data.val['pipeline'].append({})
cfg.data.val['pipeline'][2] = cfg.data.val['pipeline'][1]
cfg.data.val['pipeline'][1] =\
    {'type': 'LoadAnnotations', 'reduce_zero_label': False, 'suppress_labels': []}

## Make sure it collects gt semantic seg
cfg.data.val['pipeline'][2]['transforms'][3]['keys'].append('gt_semantic_seg')
cfg.data.val['pipeline'][2]['transforms'][4]['keys'].append('gt_semantic_seg')

# Remove some augmentationsfrom val pipeline
#cfg.data.val['pipeline'][2]['transforms'][1] = {'type': 'RandomFlip', 'prob': 0.0}
#cfg.data.val['pipeline'][2]['transforms'][2] = {'type': 'Resize', 'img_scale': (512, 512), 'ratio_range': (1.0, 1.0)}


dataset = build_dataset(cfg.data.val)
# More changes to switch to RGB-S data
dataset.img_suffix = '.npy'
dataset.seg_map_suffix = '.npy'

distributed = False

data_loader = build_dataloader(
    dataset,
    samples_per_gpu=1,
    workers_per_gpu=cfg.data.workers_per_gpu,
    dist=distributed,
    shuffle=False)

# Change model parameters to run deeplabv3 base (not maskclip + version)
RGBS_NUM_CLASSES = 2 # background + foreground
# Testing 21 to see if it fixes it

#cfg.model['pretrained'] = 'open-mmlab://resnet50_v1c'
# By deafult pretrained set twice

del cfg.model['pretrained']
cfg.model['backbone'] = {'type': 'ResNetV1c', 'depth': 50, 'num_stages': 4, \
    'out_indices': (0, 1, 2, 3), 'dilations': (1, 1, 2, 4), 'strides': (1, 2, 1, 1),\
            'norm_cfg': {'type': 'SyncBN', 'requires_grad': True}, 'norm_eval': False,\
                'style': 'pytorch', 'contract_dilation': True, 'pretrained': 'open-mmlab://resnet50_v1c'}

cfg.model['decode_head'] = {'type': 'DepthwiseSeparableASPPHead', 'in_channels': 2048,\
    'in_index': 3, 'channels': 512, 'dilations': (1, 12, 24, 36), 'c1_in_channels': 256,\
            'c1_channels': 48, 'dropout_ratio': 0.1, 'num_classes': RGBS_NUM_CLASSES,\
                'norm_cfg': {'type': 'SyncBN', 'requires_grad': True}, \
                'align_corners': False, 'loss_decode': {'type': 'CrossEntropyLoss', 'use_sigmoid': False, 'loss_weight': 1.0}}

cfg.model['auxiliary_head'] = {'type': 'FCNHead', 'in_channels': 1024, 'in_index': 2, 'channels': 256, 'num_convs': 1,\
        'concat_input': False, 'dropout_ratio': 0.1, 'num_classes': RGBS_NUM_CLASSES, 'norm_cfg': {'type': 'SyncBN', 'requires_grad': True},\
            'align_corners': False, 'loss_decode': {'type': 'CrossEntropyLoss', 'use_sigmoid': False, 'loss_weight': 0.4}}

####
# Build the detector
model = build_segmentor(cfg.model)
# Add an attribute for visualization convenience
model.CLASSES = dataset.CLASSES

# Update model to accept 4CH inputs
model.backbone.stem[0] = nn.Conv2d(4, 32, kernel_size=(3,3), stride=(2,2), padding=(1,1))
model.to(torch.device('cuda'))

checkpoint = load_checkpoint(model, args.load_from, map_location='cpu')

# SyncBN is not support for DP
model = revert_sync_batchnorm(model)

# Create work_dir
# Set up working dir to save files and logs.
cfg.work_dir = './work_dirs/vis_trained_model'
cfg.gpu_ids = range(1)
cfg.seed = 5050
#cfg.device = get_device()

mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
#train_segmentor(model, datasets, cfg, distributed=False, validate=True, 
#                meta=dict())

# We want to run test eval here on trained model

results = single_gpu_test(
    model,
    ##
    data_loader,
    True,
    args.show_dir,
    False,
    args.opacity,
    pre_eval=True,
    format_only=False,
    # To change imsave to the individual tensor entries for 4D maps
    # +2DO: check the scaling of the logits as well
    produce_maskclip_maps=False)
