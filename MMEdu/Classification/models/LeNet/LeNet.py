model = dict(
    type='ImageClassifier',
    backbone=dict(type='LeNet5', num_classes=10),
    neck=None,
    head=dict(
        type='ClsHead', loss=dict(type='CrossEntropyLoss', loss_weight=1.0)))
dataset_type = 'MNIST'
data_preprocessor = dict(mean=[33.46], std=[78.87], num_classes=10)
pipeline = [dict(type='Resize', scale=32), dict(type='PackClsInputs')]
common_data_cfg = dict(
    type='MNIST',
    data_prefix='data/mnist',
    pipeline=[dict(type='Resize', scale=32),
              dict(type='PackClsInputs')])
train_dataloader = dict(
    batch_size=128,
    num_workers=2,
    dataset=dict(
        type='MNIST',
        data_prefix='data/mnist',
        pipeline=[dict(type='Resize', scale=32),
                  dict(type='PackClsInputs')],
        test_mode=False),
    sampler=dict(type='DefaultSampler', shuffle=True))
val_dataloader = dict(
    batch_size=128,
    num_workers=2,
    dataset=dict(
        type='MNIST',
        data_prefix='data/mnist',
        pipeline=[dict(type='Resize', scale=32),
                  dict(type='PackClsInputs')],
        test_mode=True),
    sampler=dict(type='DefaultSampler', shuffle=False))
val_evaluator = dict(type='Accuracy', topk=(1, ))
test_dataloader = dict(
    batch_size=128,
    num_workers=2,
    dataset=dict(
        type='MNIST',
        data_prefix='data/mnist',
        pipeline=[dict(type='Resize', scale=32),
                  dict(type='PackClsInputs')],
        test_mode=True),
    sampler=dict(type='DefaultSampler', shuffle=False))
test_evaluator = dict(type='Accuracy', topk=(1, ))
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[15], gamma=0.1)
train_cfg = dict(by_epoch=True, max_epochs=5, val_interval=1)
val_cfg = dict()
test_cfg = dict()
default_scope = 'mmcls'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=150),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
log_level = 'INFO'
load_from = None
resume_from = None
auto_scale_lr = dict(base_batch_size=128)
