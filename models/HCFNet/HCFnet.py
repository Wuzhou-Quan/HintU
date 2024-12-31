import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModel
from torch.nn import BCELoss
from ..loss import GeneralizedWassersteinDiceLoss, SoftLoULoss
from ..focalloss import FocalLoss

# from ptflops import get_model_complexity_info
from .main_blocks import PPA, DASI, MDCR


class HCFNet(BaseModel):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
    ) -> None:
        super().__init__()
        self._hcfnet = hcf_net(in_channels, out_channels)
        self.loss1 = BCELoss()
        self.loss1_1 = FocalLoss()
        self.loss2 = BCELoss()
        self.loss2_1 = FocalLoss()
        self.loss3 = BCELoss()
        self.loss3_1 = FocalLoss()
        self.loss4 = BCELoss()
        self.loss4_1 = FocalLoss()
        self.loss5 = BCELoss()
        self.loss5_1 = FocalLoss()

    def forward(self, x, gt=None, mode="predict"):
        out, mout1, mout2, mout3, mout4 = self._hcfnet(x)
        if mode == "predict":
            return out
        else:
            return {
                "loss": (
                    (self.loss4(mout4, gt) + self.loss4_1(mout4, gt)) * 0.0625
                    + (self.loss3(mout3, gt) + self.loss3_1(mout3, gt)) * 0.125
                    + (self.loss2(mout2, gt) + self.loss2_1(mout2, gt)) * 0.25
                    + (self.loss1(mout1, gt) + self.loss1_1(mout1, gt)) * 0.5
                    + (self.loss5(out, gt) + self.loss5_1(out, gt))
                )
                * 10000
            }


class hcf_net(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
    ) -> None:
        super().__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.p1 = PPA(in_features=in_channels, filters=32)

        self.respath1 = DASI(
            in_features=32,
            out_features=32,
        )
        self.p2 = PPA(in_features=32, filters=int(32 * 2))
        self.respath2 = DASI(
            in_features=64,
            out_features=32 * 2,
        )
        self.p3 = PPA(in_features=64, filters=int(32 * 4))
        self.respath3 = DASI(
            in_features=128,
            out_features=32 * 4,
        )
        self.p4 = PPA(in_features=128, filters=int(32 * 8))
        self.respath4 = DASI(
            in_features=256,
            out_features=32 * 8,
        )
        self.p5 = PPA(in_features=256, filters=int(32 * 16))

        self.mdcr = MDCR(in_features=int(512), out_features=int(512))

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 32 * 8, kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(32 * 8),
            nn.ReLU(),
        )

        self.p6 = PPA(in_features=32 * 8 * 2, filters=int(32 * 8))
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 32 * 4, kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(32 * 4),
            nn.ReLU(),
        )

        self.p7 = PPA(in_features=32 * 4 * 2, filters=int(32 * 4))

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 32 * 2, kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(32 * 2),
            nn.ReLU(),
        )

        self.p8 = PPA(in_features=32 * 2 * 2, filters=int(32 * 2))
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.p9 = PPA(in_features=32 * 2, filters=int(32))

        self.out = nn.Conv2d(
            in_channels=32,
            out_channels=out_channels,
            kernel_size=(1, 1),
            padding=(0, 0),
        )
        self.out1 = nn.Conv2d(
            in_channels=512,
            out_channels=out_channels,
            kernel_size=(1, 1),
            padding=(0, 0),
        )
        self.out2 = nn.Conv2d(
            in_channels=256,
            out_channels=out_channels,
            kernel_size=(1, 1),
            padding=(0, 0),
        )
        self.out3 = nn.Conv2d(
            in_channels=128,
            out_channels=out_channels,
            kernel_size=(1, 1),
            padding=(0, 0),
        )
        self.out4 = nn.Conv2d(
            in_channels=64,
            out_channels=out_channels,
            kernel_size=(1, 1),
            padding=(0, 0),
        )

    def forward(self, x):
        # encoder
        x1 = self.p1(x)
        xp1 = self.maxpool(x1)
        x2 = self.p2(xp1)
        xp2 = self.maxpool(x2)
        x3 = self.p3(xp2)
        xp3 = self.maxpool(x3)
        x4 = self.p4(xp3)
        xp4 = self.maxpool(x4)
        x = self.p5(xp4)
        x = self.mdcr(x)

        x1_res = self.respath1(x1, x2, None)  # 1 32 512 512
        x2_res = self.respath2(x2, x3, x1)  # 1 64 256 256
        x3_res = self.respath3(x3, x4, x2)  # 1 128 128 128
        x4_res = self.respath4(x4, x, x3)  # 1 256 64 64

        # decoder
        out4 = F.interpolate(
            self.out1(x), scale_factor=16, mode="bilinear", align_corners=True
        )
        x = self.up1(x)
        x = torch.cat((x, x4_res), dim=1)
        x = self.p6(x)
        out3 = F.interpolate(
            self.out2(x), scale_factor=8, mode="bilinear", align_corners=True
        )
        x = self.up2(x)
        x = torch.cat((x, x3_res), dim=1)
        x = self.p7(x)
        out2 = F.interpolate(
            self.out3(x), scale_factor=4, mode="bilinear", align_corners=True
        )
        x = self.up3(x)
        x = torch.cat((x, x2_res), dim=1)
        x = self.p8(x)
        out1 = F.interpolate(
            self.out4(x), scale_factor=2, mode="bilinear", align_corners=True
        )
        x = self.up4(x)
        x = torch.cat((x, x1_res), dim=1)
        x = self.p9(x)
        out = torch.sigmoid(self.out(x))

        return (
            out,
            torch.sigmoid(out1),
            torch.sigmoid(out2),
            torch.sigmoid(out3),
            torch.sigmoid(out4),
        )
