from models.UNet.nestedunet import NestedUNet1024C

model = dict(type=NestedUNet1024C, in_channels=1, out_channels=1)