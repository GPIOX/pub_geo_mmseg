# dataset settings
dataset_type = 'GeologicalHazard'
data_root = 'data5'
# data_root = 'data6'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        ratio_range=(0.5, 2.0),
        scale=(512, 512),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        ann_file='train_3.txt',
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        ann_file='test_3.txt',
        pipeline=test_pipeline))
test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# data = dict(
#     samples_per_gpu=4,
#     workers_per_gpu=1,
#     train=dict(
#         type=dataset_type,
#         data_root=data_root,
#         img_dir='JPEGImages',
#         ann_dir='SegmentationClass',
#         split='train_3.txt',
#         pipeline=train_pipeline),
#     val=dict(
#         type=dataset_type,
#         data_root=data_root,
#         img_dir='JPEGImages',
#         ann_dir='SegmentationClass',
#         split='val_3.txt',
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         data_root=data_root,
#         img_dir='JPEGImages',
#         ann_dir='SegmentationClass',
#         split='test_3.txt',
#         pipeline=test_pipeline))
