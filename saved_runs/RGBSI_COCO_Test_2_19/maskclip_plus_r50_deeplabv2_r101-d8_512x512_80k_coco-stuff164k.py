norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='MaskClipPlusHead',
        in_channels=2048,
        channels=1024,
        num_classes=0,
        dropout_ratio=0,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        decode_module_cfg=dict(
            type='ASPPHeadV2', input_transform=None,
            dilations=(6, 12, 18, 24)),
        text_categories=171,
        text_channels=1024,
        text_embeddings_path='pretrain/coco_RN50_clip_text.pth',
        cls_bg=False,
        norm_feat=False,
        clip_unlabeled_cats=[
            19, 23, 28, 29, 36, 51, 76, 88, 94, 112, 133, 136, 137, 157, 160
        ],
        clip_cfg=dict(
            type='ResNetClip',
            depth=50,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            contract_dilation=True),
        clip_weights_path='pretrain/RN50_clip_weights.pth',
        reset_counter=True,
        start_clip_guided=(1, 7999),
        start_self_train=(8000, -1),
        unlabeled_cats=[
            19, 23, 28, 29, 36, 51, 76, 88, 94, 112, 133, 136, 137, 157, 160
        ]),
    feed_img_to_decode_head=True,
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'COCOStuffDataset'
data_root = 'data/coco'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations',
        suppress_labels=[
            19, 23, 28, 29, 36, 51, 76, 88, 94, 112, 133, 136, 137, 157, 160
        ]),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='COCOStuffDataset',
        data_root='data/coco',
        img_dir='images/train2017',
        ann_dir='annotations/train2017',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='LoadAnnotations',
                suppress_labels=[
                    19, 23, 28, 29, 36, 51, 76, 88, 94, 112, 133, 136, 137,
                    157, 160
                ]),
            dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='COCOStuffDataset',
        data_root='data/coco',
        img_dir='images/val2017',
        ann_dir='annotations/val2017',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='COCOStuffDataset',
        data_root='data/coco',
        img_dir='images/val2017',
        ann_dir='annotations/val2017',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=8000)
evaluation = dict(interval=8000, metric='mIoU', pre_eval=True)
suppress_labels = [
    19, 23, 28, 29, 36, 51, 76, 88, 94, 112, 133, 136, 137, 157, 160
]
find_unused_parameters = True
work_dir = 'saved_runs/RGBSI_COCO_Test_2_19'
gpu_ids = range(0, 4)
auto_resume = False
