from models.focalloss import FocalLoss
from models.MIMISTD.mimistd_plus import MiM_plus

model = dict(
    type=MiM_plus,
    layer_blocks=[2] * 3,
    channels=[8, 16, 32, 64, 128],
    loss=FocalLoss,
)
