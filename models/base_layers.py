import torch.nn as nn


class CNA1x1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CNA1x1, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1)
        self.norm1 = nn.BatchNorm2d(out_ch)
        self.act1 = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.conv1(input)
        x = self.norm1(x)
        x = self.act1(x)
        return x


class CNA3x3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CNA3x3, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_ch)
        self.act1 = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.conv1(input)
        x = self.norm1(x)
        x = self.act1(x)
        return x


class DoubleCNA3x3(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super(DoubleCNA3x3, self).__init__()
        if mid_ch is None:
            mid_ch = out_ch
        self.CNA1 = CNA3x3(in_ch, mid_ch)
        self.CNA2 = CNA3x3(mid_ch, out_ch)
        self.outproj = nn.Conv2d(out_ch, out_ch, 1)

    def forward(self, input):
        x = self.CNA1(input)
        x = self.CNA2(x)
        x = self.outproj(x)
        return x


class ResidualDoubleCNA3x3(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None):
        super(ResidualDoubleCNA3x3, self).__init__()
        if mid_ch is None:
            mid_ch = out_ch
        self.double_cna = DoubleCNA3x3(in_ch, out_ch, mid_ch)
        self.align = CNA1x1(in_ch, out_ch) if in_ch != out_ch else nn.Identity()
        self.outproj = nn.Conv2d(out_ch, out_ch, 1)

    def forward(self, input):
        x = self.double_cna(input)
        x = x + self.align(input)
        x = self.outproj(x)
        return x


class MLP(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_ch, mid_ch=None, out_ch=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_ch = out_ch or in_ch
        mid_ch = mid_ch or in_ch
        self.fc1 = nn.Conv2d(in_ch, mid_ch, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(mid_ch, out_ch, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
