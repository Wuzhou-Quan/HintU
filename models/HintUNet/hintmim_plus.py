from ..MIMISTD.mimistd_plus import MiM_plus
from .hint import Hint
from mmengine.model import BaseModel
import torch
from ..base_layers import DoubleCNA3x3, CNA3x3


class HintMiMPlus(BaseModel):
    def __init__(self, layer_blocks, channels, in_channels: int, out_channels: int, c_base: int = 32, loss = None):
        super().__init__()
        self.hint = Hint(1, c_base, c_base * 4)
        self.x_proj = CNA3x3(in_channels, c_base)
        self.hintx_proj = DoubleCNA3x3(c_base * 2, c_base, c_base // 2)
        self.mim = MiM_plus(layer_blocks, channels, c_base , img_size=256)
        self.loss = loss()

    def forward(self, x, gt=None, mode="predict"):
        hint = self.hint(x)
        x = self.x_proj(x)
        x = torch.cat([x, hint], dim=1)
        x = self.hintx_proj(x)
        x = self.mim(x)

        if mode == "predict":
            return x
        else:
            return {"loss": self.loss(x, gt) * 10000}
