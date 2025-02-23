_base_ = [
    '../_base_/models/pspnet_vit16.py',
    '../_base_/datasets/pascal_context_59.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
model = dict(
    decode_head=dict(num_classes=59)
)
optimizer = dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001)
# data = dict(
#     samples_per_gpu=2,
# )