_base_ = [
    './models/fast_scnn.py', './datasets/trash.py',
    './default_runtime.py', './schedules/schedule_160k.py'
]

# Re-config the data sampler.
data = dict(samples_per_gpu=4, workers_per_gpu=4)

# Re-config the optimizer.
optimizer = dict(type='SGD', lr=0.12, momentum=0.9, weight_decay=4e-5)
