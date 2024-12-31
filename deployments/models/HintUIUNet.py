from models.HintUNet.hint_unet_framework import GenericHintU
from models.UNet.UIUNet import UIUNET

model = dict(
    type=GenericHintU, in_channels=1, out_channels=1, unet_cons=UIUNET, c_base=32
)
