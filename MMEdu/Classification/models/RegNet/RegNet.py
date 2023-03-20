model = dict(
    type='ImageClassifier',
    backbone=dict(type='RegNet', arch='regnetx_3.2gf'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1008,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)))
dataset_type = 'ImageNet'
data_preprocessor = dict(
    num_classes=1000,
    mean=[103.53, 116.28, 123.675],
    std=[57.375, 57.12, 58.395],
    to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='Lighting',
        eigval=[0.2175, 0.0188, 0.0045],
        eigvec=[[-0.5836, -0.6948, 0.4203], [-0.5808, -0.0045, -0.814],
                [-0.5675, 0.7192, 0.4009]],
        alphastd=25.5,
        to_rgb=False),
    dict(type='PackClsInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackClsInputs')
]
train_dataloader = dict(
    batch_size=64,
    num_workers=5,
    dataset=dict(
        type='ImageNet',
        data_root='data/imagenet',
        ann_file='meta/train.txt',
        data_prefix='train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='RandomResizedCrop', scale=224),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(
                type='Lighting',
                eigval=[0.2175, 0.0188, 0.0045],
                eigvec=[[-0.5836, -0.6948, 0.4203], [-0.5808, -0.0045, -0.814],
                        [-0.5675, 0.7192, 0.4009]],
                alphastd=25.5,
                to_rgb=False),
            dict(type='PackClsInputs')
        ]),
    sampler=dict(type='DefaultSampler', shuffle=True))
val_dataloader = dict(
    batch_size=128,
    num_workers=5,
    dataset=dict(
        type='ImageNet',
        data_root='data/imagenet',
        ann_file='meta/val.txt',
        data_prefix='val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='ResizeEdge', scale=256, edge='short'),
            dict(type='CenterCrop', crop_size=224),
            dict(type='PackClsInputs')
        ]),
    sampler=dict(type='DefaultSampler', shuffle=False))
val_evaluator = dict(type='Accuracy', topk=(1, 5))
test_dataloader = dict(
    batch_size=128,
    num_workers=5,
    dataset=dict(
        type='ImageNet',
        data_root='data/imagenet',
        ann_file='meta/val.txt',
        data_prefix='val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='ResizeEdge', scale=256, edge='short'),
            dict(type='CenterCrop', crop_size=224),
            dict(type='PackClsInputs')
        ]),
    sampler=dict(type='DefaultSampler', shuffle=False))
test_evaluator = dict(type='Accuracy', topk=(1, 5))
optim_wrapper = dict(
    optimizer=dict(
        type='SGD', lr=0.4, momentum=0.9, weight_decay=5e-05, nesterov=True))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=5),
    dict(type='CosineAnnealingLR', T_max=95, by_epoch=True, begin=5, end=100)
]
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()
test_cfg = dict()
auto_scale_lr = dict(base_batch_size=512)
default_scope = 'mmcls'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='VisualizationHook', enable=False))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='ClsVisualizer', vis_backends=[dict(type='LocalVisBackend')])
log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=None, deterministic=False)
EIGVAL = [0.2175, 0.0188, 0.0045]
EIGVEC = [[-0.5836, -0.6948, 0.4203], [-0.5808, -0.0045, -0.814],
          [-0.5675, 0.7192, 0.4009]]
custom_hooks = [
    dict(
        type='PreciseBNHook',
        num_samples=8192,
        interval=1,
        priority='ABOVE_NORMAL')
]
