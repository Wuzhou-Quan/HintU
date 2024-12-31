import torch, einops
from torch import nn, Tensor
from torch.nn import functional as F
from .netutils import fill_fc_weights


def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    temp = query.bmm(key.transpose(1, 2))
    scale = query.size(-1) ** 0.5
    softmax = F.softmax(temp / scale, dim=-1)
    return softmax.bmm(value)
    # temp = torch.einsum("bqd,bkd->bqk", query, key)
    # scale = query.size(-1) ** 0.5
    # attention_weights = F.softmax(temp / scale, dim=-1)
    # return torch.einsum("bqk,bkv->bqv", attention_weights, value)


class HeadAttention(nn.Module):
    def __init__(self, dim_in: int, dim_k: int, dim_v: int):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_k)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_v)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return scaled_dot_product_attention(self.q(query), self.k(key), self.v(value))


class CrossAttention(nn.Module):
    def __init__(self, qv_in_channels, k_in_channels):
        super(CrossAttention, self).__init__()
        self.conv_query = nn.Conv2d(in_channels=qv_in_channels, out_channels=qv_in_channels, kernel_size=1, bias=True)
        self.conv_key = nn.Conv2d(in_channels=k_in_channels, out_channels=k_in_channels, kernel_size=1, bias=True)
        self.conv_value = nn.Conv2d(in_channels=qv_in_channels, out_channels=k_in_channels, kernel_size=1, bias=True)
        self.conv_out = nn.Sequential(
            nn.BatchNorm2d(qv_in_channels, eps=0.0001, momentum=0.95),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=qv_in_channels, out_channels=qv_in_channels, kernel_size=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=qv_in_channels, out_channels=qv_in_channels, kernel_size=1, bias=True),
        )
        fill_fc_weights(self)

    def forward(self, q, v):
        query = self.conv_query(q)
        key = self.conv_key(v)
        value = self.conv_value(q)

        query = query.view(query.size(0), query.size(1), -1)
        key = key.view(key.size(0), key.size(1), -1)
        value = value.view(value.size(0), value.size(1), -1)

        out = scaled_dot_product_attention(query, key, value)

        out = out.permute(0, 2, 1).contiguous()
        out = out.view(q.size(0), -1, q.size(2), q.size(3))
        out = self.conv_out(out)

        return out


class SelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SelfAttention, self).__init__()
        self.conv_query = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.conv_key = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.conv_value = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.conv_out = nn.Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=1)

    def forward(self, x):
        query = self.conv_query(x)
        key = self.conv_key(x)
        value = self.conv_value(x)

        query = query.view(query.size(0), query.size(1), -1)
        key = key.view(key.size(0), key.size(1), -1)
        value = value.view(value.size(0), value.size(1), -1)

        out = scaled_dot_product_attention(query, key, value)

        out = out.permute(0, 2, 1).contiguous()
        out = out.view(x.size(0), -1, x.size(2), x.size(3))
        out = self.conv_out(out)

        return out


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention"""

    def __init__(self, n_heads, q_in_channels, k_in_channels, v_in_channels):
        super().__init__()

        self.n_head = n_heads
        self.conv_query = nn.Conv2d(in_channels=q_in_channels, out_channels=q_in_channels * n_heads, kernel_size=1, bias=True)
        self.conv_key = nn.Conv2d(in_channels=k_in_channels, out_channels=v_in_channels * n_heads, kernel_size=1, bias=True)
        self.conv_value = nn.Conv2d(in_channels=v_in_channels, out_channels=v_in_channels * n_heads, kernel_size=1, bias=True)

        self.conv_out = nn.Sequential(
            nn.BatchNorm2d(v_in_channels * n_heads, eps=0.0001, momentum=0.95),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=v_in_channels * n_heads, out_channels=v_in_channels * n_heads, kernel_size=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=v_in_channels * n_heads, out_channels=v_in_channels, kernel_size=1, bias=True),
        )

    def forward(self, q, k, v):
        query = self.conv_query(q)
        key = self.conv_key(k)
        value = self.conv_value(v)

        query = query.view(query.size(0), query.size(1), -1)
        key = key.view(key.size(0), key.size(1), -1)
        value = value.view(value.size(0), value.size(1), -1)

        out = scaled_dot_product_attention(query, key, value)

        out = out.permute(0, 2, 1).contiguous()
        out = out.view(v.size(0), -1, v.size(2), v.size(3))
        out = self.conv_out(out)

        return out


class MultiHeadAttention_SimilarKV(MultiHeadAttention):
    """Multi-Head Attention"""

    def __init__(self, n_heads, kv_in_channels, q_in_channels):
        super().__init__(n_heads, kv_in_channels, q_in_channels, kv_in_channels)

    def forward(self, kv, q):
        return super().forward(kv, q, kv)


class MultiHeadSelfAttention(MultiHeadAttention):
    def __init__(self, n_heads, channels):
        super().__init__(n_heads, channels, channels, channels)

    def forward(self, obj):
        return super().forward(obj, obj, obj)
