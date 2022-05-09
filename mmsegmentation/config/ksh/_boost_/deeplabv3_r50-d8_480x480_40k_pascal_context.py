_base_ = [
    './models/deeplabv3_r50-d8.py', './datasets/trash.py',
    './default_runtime.py', './schedules/schedule_160k.py'
]
model = dict(
    decode_head=dict(num_classes=11),
    auxiliary_head=dict(num_classes=11),
    test_cfg=dict(mode='slide', crop_size=(480, 480), stride=(320, 320)))
optimizer = dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001)
