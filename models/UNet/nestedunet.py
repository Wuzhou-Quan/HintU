# Code based on repository: https://github.com/Andy-zhujunwen/UNET-ZOO

import torch
from torch import nn
from mmengine.model import BaseModel
from torch.nn import MSELoss, BCELoss
from ..loss import GeneralizedWassersteinDiceLoss, SoftLoULoss
from ..focalloss import FocalLoss


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.conv(input)


class NestedUNet1024C(BaseModel):
    def __init__(self, in_channels, out_channels, nb_filter=[64, 128, 256, 512, 1024]):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv0_0 = DoubleConv(in_channels, nb_filter[0])
        self.conv1_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv2_0 = DoubleConv(nb_filter[1], nb_filter[2])
        self.conv3_0 = DoubleConv(nb_filter[2], nb_filter[3])
        self.conv4_0 = DoubleConv(nb_filter[3], nb_filter[4])

        self.conv0_1 = DoubleConv(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = DoubleConv(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv2_1 = DoubleConv(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv3_1 = DoubleConv(nb_filter[3] + nb_filter[4], nb_filter[3])

        self.conv0_2 = DoubleConv(nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = DoubleConv(nb_filter[1] * 2 + nb_filter[2], nb_filter[1])
        self.conv2_2 = DoubleConv(nb_filter[2] * 2 + nb_filter[3], nb_filter[2])

        self.conv0_3 = DoubleConv(nb_filter[0] * 3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = DoubleConv(nb_filter[1] * 3 + nb_filter[2], nb_filter[1])

        self.conv0_4 = DoubleConv(nb_filter[0] * 4 + nb_filter[1], nb_filter[0])
        self.final = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
        self.loss1 = BCELoss()
        self.loss2 = FocalLoss()

    def forward(self, x, gt=None, mode="predict"):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        upx1_0 = self.up(x1_0)
        x0_1 = self.conv0_1(torch.cat([x0_0, upx1_0], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        output = self.final(x0_4).sigmoid()

        if mode == "predict":
            return output
        else:
            return {"loss": (self.loss1(output, gt) + self.loss2(output, gt)) * 10000}


class NestedUNet(BaseModel):
    def __init__(self, in_channels, out_channels, nb_filter=[32, 64, 128, 256, 512]):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv0_0 = DoubleConv(in_channels, nb_filter[0])
        self.conv1_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv2_0 = DoubleConv(nb_filter[1], nb_filter[2])
        self.conv3_0 = DoubleConv(nb_filter[2], nb_filter[3])
        self.conv4_0 = DoubleConv(nb_filter[3], nb_filter[4])

        self.conv0_1 = DoubleConv(nb_filter[0] + nb_filter[1], nb_filter[0])
        self.conv1_1 = DoubleConv(nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv2_1 = DoubleConv(nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv3_1 = DoubleConv(nb_filter[3] + nb_filter[4], nb_filter[3])

        self.conv0_2 = DoubleConv(nb_filter[0] * 2 + nb_filter[1], nb_filter[0])
        self.conv1_2 = DoubleConv(nb_filter[1] * 2 + nb_filter[2], nb_filter[1])
        self.conv2_2 = DoubleConv(nb_filter[2] * 2 + nb_filter[3], nb_filter[2])

        self.conv0_3 = DoubleConv(nb_filter[0] * 3 + nb_filter[1], nb_filter[0])
        self.conv1_3 = DoubleConv(nb_filter[1] * 3 + nb_filter[2], nb_filter[1])

        self.conv0_4 = DoubleConv(nb_filter[0] * 4 + nb_filter[1], nb_filter[0])
        self.final = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
        self.loss1 = BCELoss()
        self.loss2 = FocalLoss()

    def forward(self, x, gt=None, mode="predict"):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        upx1_0 = self.up(x1_0)
        x0_1 = self.conv0_1(torch.cat([x0_0, upx1_0], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        output = self.final(x0_4).sigmoid()

        if mode == "predict":
            return output
        else:
            return {"loss": (self.loss1(output, gt) + self.loss2(output, gt)) * 10000}
