from models.HintUNet.hint_unet_framework import GenericHintU
from models.UNet.nestedunet import NestedUNet1024C

model = dict(type=GenericHintU, in_channels=1, out_channels=1, unet_cons=NestedUNet1024C)
