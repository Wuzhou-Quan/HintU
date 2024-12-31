from models.UNet.nestedunet import NestedUNet

model = dict(type=NestedUNet, in_channels=1, out_channels=1)