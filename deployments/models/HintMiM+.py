from models.focalloss import FocalLoss
from models.HintUNet.hintmim_plus import HintMiMPlus

model = dict(type=HintMiMPlus, layer_blocks=[2] * 3, channels=[8, 16, 32, 64, 128], loss = FocalLoss, in_channels=1, out_channels=1)
