# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings
import sys
#sys.path.append('/home/nwarner30/Insync/nikolaiwarner7@gmail.com/OneDrive/Spring 2023/Research/MaskCLIP')
#sys.path.append('/home/nwarner30/Insync/nikolaiwarner7@gmail.com/OneDrive/Spring 2023/Research/MaskCLIP/mmseg')
sys.path.append('/root/MaskCLIP/')
sys.path.append('/root/MaskCLIP/mmseg')

import mmcv
import torch
import torch.nn as nn
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import Config, DictAction, get_git_hash

from mmseg import __version__
from mmseg.apis import init_random_seed, set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')

    # Switch to use deeplab backbone
    #parser.add_argument('--config', default='configs/deeplabv3plus/deeplabv3plus_r50-d8_512x512_40k_voc12aug.py', help='test config file path')
    parser.add_argument('--config', default='configs/maskclip_plus/zero_shot/maskclip_plus_r50_deeplabv2_r101-d8_512x512_20k_voc12aug_20.py', help='test config file path')
    parser.add_argument('--work-dir', default = 'test_outs/', help='the dir to save logs and models')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')
    parser.add_argument(
        '--resume-from', default='test_outs/latest.pth', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=5050, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help="--options is deprecated in favor of --cfg_options' and it will "
        'not be supported in version v0.22.0. Override some settings in the '
        'used config, the key-value pair in xxx=yyy format will be merged '
        'into config file. If the value to be overwritten is a list, it '
        'should be like key="[a,b]" or key=a,b It also allows nested '
        'list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation '
        'marks are necessary and that no white space is allowed.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        default = True, # Resume training from last point
        help='resume from the latest checkpoint automatically.')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options. '
            '--options will not be supported in version v0.22.0.')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options. '
                      '--options will not be supported in version v0.22.0.')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
    cfg.auto_resume = args.auto_resume

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # gpu_ids is used to calculate iter when resuming checkpoint
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    seed = init_random_seed(args.seed)
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

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
    
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
 
    # SyncBN is not support for DP
    if not distributed:
        warnings.warn(
            'SyncBN is only supported with DDP. To be compatible with DP, '
            'we convert SyncBN to BN. Please use dist_train.sh which can '
            'avoid this error.')
        model = revert_sync_batchnorm(model)

    logger.info(model)

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
    cfg.data['samples_per_gpu'] = 12

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
    

    datasets = [build_dataset(cfg.data.train)]
    # More changes to switch to RGB-S data
    datasets[0].img_suffix = '.npy'
    datasets[0].seg_map_suffix = '.npy'

    # Set the number of training steps (iterations)
    cfg.runner['max_iters'] = 3e5
    
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmseg version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    # passing checkpoint meta for saving best checkpoint
    meta.update(cfg.checkpoint_config.meta)

    # Update model to accept 4CH inputs
    model.backbone.stem[0] = nn.Conv2d(4, 32, kernel_size=(3,3), stride=(2,2), padding=(1,1))
    model.to(torch.device('cuda'))

    # For debugging call validation interval sooner
    EVAL_INTERVAL = 2000
    cfg.checkpoint_config.interval = EVAL_INTERVAL
    cfg.evaluation.interval = EVAL_INTERVAL

    cfg.optimizer['lr'] = 4.8e-3

    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    os.chdir('/root/MaskCLIP/')
    #os.chdir('/home/nwarner30/Insync/nikolaiwarner7@gmail.com/OneDrive/Spring 2023/Research/MaskCLIP')
    main()
