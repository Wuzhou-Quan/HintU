import torch
from mmengine.model import BaseModel
from torch.nn import BCELoss
from .hint import Hint
from ..focalloss import FocalLoss
from ..base_layers import DoubleCNA3x3, CNA3x3
from ..UNet.unet import UNet


class GenericHintU(BaseModel):
    def __init__(self, in_channels: int, out_channels: int, unet_cons=UNet, c_base: int = 32):
        super().__init__()
        self.hint = Hint(in_channels, c_base, c_base * 4)
        self.x_proj = CNA3x3(in_channels, c_base)
        self.hintx_proj = DoubleCNA3x3(c_base * 2, c_base, c_base // 2)
        self.unet = unet_cons(c_base, out_channels)
        self.loss1 = BCELoss()
        self.loss2 = FocalLoss()

    def forward(self, x, gt=None, mode="predict"):
        hint = self.hint(x)
        x_p = self.x_proj(x)
        x = torch.cat([x_p, hint], dim=1)
        hintx = self.hintx_proj(x)
        ret = self.unet(hintx)

        if mode == "predict":
            return ret
        elif mode == "predict_vis":
            return x, hint, hintx, ret
        else:
            return {"loss": (self.loss1(ret, gt) + self.loss2(ret, gt)) * 10000}
