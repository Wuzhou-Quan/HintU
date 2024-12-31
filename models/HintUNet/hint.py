import torch
from ..base_layers import DoubleCNA3x3, CNA3x3, MLP
from ..attentions import CrossAttention


class Hint(torch.nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        if mid_ch == None:
            mid_ch = out_ch

        avgpool_k = 3
        self.avgpool_padding = (avgpool_k - 1) // 2
        self.avgpool = torch.nn.AvgPool2d(avgpool_k, 1, padding=self.avgpool_padding)

        maxpool_k = 3
        self.maxpool_padding = (maxpool_k - 1) // 2
        self.maxpool = torch.nn.MaxPool2d(maxpool_k, 1, padding=self.maxpool_padding)

        self.x_proj = CNA3x3(in_ch, mid_ch)
        self.hint_proj = DoubleCNA3x3(in_ch * 2, mid_ch)
        self.cross_attn = CrossAttention(mid_ch, mid_ch)
        self.mlp_fnorm = torch.nn.BatchNorm2d(mid_ch)
        self.mlp = MLP(mid_ch, mid_ch * 2, out_ch)

    def forward(self, x):
        avgpooling = self.avgpool(x.detach())
        maxpooling = self.maxpool(x.detach())
        hint = self.hint_proj(torch.cat([x - avgpooling, x - maxpooling], dim=1))
        x_d = self.x_proj(x)
        attn = self.cross_attn(x_d, hint)
        x_d = self.mlp_fnorm(x_d + attn)
        mlp_res = self.mlp(x_d)
        return mlp_res


class IHint(torch.nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super().__init__()
        if mid_ch == None:
            mid_ch = out_ch

        avgpool_k = 3
        self.avgpool_padding = (avgpool_k - 1) // 2
        self.avgpool = torch.nn.AvgPool2d(avgpool_k, 1, padding=self.avgpool_padding)

        maxpool_k = 3
        self.maxpool_padding = (maxpool_k - 1) // 2
        self.maxpool = torch.nn.MaxPool2d(maxpool_k, 1, padding=self.maxpool_padding)

        self.x_proj = CNA3x3(in_ch, mid_ch)
        self.hint_proj = DoubleCNA3x3(in_ch * 2, mid_ch)
        self.mlp_fnorm = torch.nn.BatchNorm2d(mid_ch)
        self.mlp = MLP(mid_ch, mid_ch * 2, out_ch)

    def forward(self, x):
        avgpooling = self.avgpool(x.detach())
        maxpooling = self.maxpool(x.detach())
        hint = self.hint_proj(torch.cat([x - avgpooling, x - maxpooling], dim=1))
        mlp_res = self.mlp(hint)
        return mlp_res
