import torch
from mmengine.model import BaseModel
from torch.nn import BCELoss
from ..HCFNet.HCFnet import hcf_net
from .hint import Hint
from ..focalloss import FocalLoss
from ..base_layers import DoubleCNA3x3, CNA3x3


class HintHCFNet(BaseModel):
    def __init__(self, in_channels: int, out_channels: int, c_base: int = 32) -> None:
        super().__init__()
        self.hint = Hint(in_channels, c_base, c_base * 4)
        self.x_proj = CNA3x3(in_channels, c_base)
        self.hintx_proj = DoubleCNA3x3(c_base * 2, c_base, c_base // 2)
        self._hcfnet = hcf_net(c_base, out_channels)
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
        hint = self.hint(x)
        x = self.x_proj(x)
        x = torch.cat([x, hint], dim=1)
        x = self.hintx_proj(x)
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
