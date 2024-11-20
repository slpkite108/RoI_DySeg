import torch.nn as nn
#from mamba_ssm import Mamba

class PatchPartition(nn.Module):
    def __init__(self, channels, lazy=False):
        super(PatchPartition, self).__init__()
        self.positional_encoding = nn.Conv3d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        ) if not lazy else nn.LazyConv3d(channels, kernel_size=3, padding=1, groups=channels, bias=False)

    def forward(self, x):
        x = self.positional_encoding(x)
        return x


class LineConv(nn.Module):
    def __init__(self, channels, lazy=False):
        super(LineConv, self).__init__()
        expansion = 4
        self.line_conv_0 = nn.Conv3d(
            channels, channels * expansion, kernel_size=1, bias=False
        ) if not lazy else nn.LazyConv3d(channels * expansion, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.line_conv_1 = nn.Conv3d(
            channels * expansion, channels, kernel_size=1, bias=False
        ) if not lazy else nn.LazyConv3d(channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.line_conv_0(x)
        x = self.act(x)
        x = self.line_conv_1(x)
        return x


class LocalRepresentationsCongregation(nn.Module):
    def __init__(self, channels, lazy=False):
        super(LocalRepresentationsCongregation, self).__init__()
        self.bn1 = nn.BatchNorm3d(channels) if not lazy else nn.LazyBatchNorm3d()
        self.pointwise_conv_0 = nn.Conv3d(channels, channels, kernel_size=1, bias=False) if not lazy else nn.LazyConv3d(channels, kernel_size=1, bias=False)
        self.depthwise_conv = nn.Conv3d(
            channels, channels, padding=1, kernel_size=3, groups=channels, bias=False
        ) if not lazy else nn.LazyConv3d(channels, padding=1, kernel_size=3, groups=channels, bias=False)
        self.bn2 = nn.BatchNorm3d(channels) if not lazy else nn.LazyBatchNorm3d()
        self.pointwise_conv_1 = nn.Conv3d(channels, channels, kernel_size=1, bias=False) if not lazy else nn.LazyConv3d(channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.bn1(x)
        x = self.pointwise_conv_0(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.pointwise_conv_1(x)
        return x


class GlobalSparseTransformer(nn.Module):
    def __init__(self, channels, r, heads, lazy=False):
        super(GlobalSparseTransformer, self).__init__()
        self.head_dim = channels // heads
        self.scale = self.head_dim**-0.5
        self.num_heads = heads
        self.sparse_sampler = nn.AvgPool3d(kernel_size=1, stride=r)
        # qkv
        self.qkv = nn.Conv3d(channels, channels * 3, kernel_size=1, bias=False) if not lazy else nn.LazyConv3d(channels * 3, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.sparse_sampler(x)
        B, C, H, W, Z = x.shape
        q, k, v = (
            self.qkv(x)
            .view(B, self.num_heads, -1, H * W * Z)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        attn = (q.transpose(-2, -1) @ k).softmax(-1)
        x = (v @ attn.transpose(-2, -1)).view(B, -1, H, W, Z)
        return x
    
    
class DiffGlobalSparseTransformer(nn.Module):
    def __init__(self, channels, r, heads, lazy=False):
        super(GlobalSparseTransformer, self).__init__()
        self.head_dim = channels // heads
        self.scale = self.head_dim**-0.5
        self.num_heads = heads
        self.sparse_sampler = nn.AvgPool3d(kernel_size=1, stride=r)
        # qkv
        self.qkv = nn.Conv3d(channels, channels * 3, kernel_size=1, bias=False) if not lazy else nn.LazyConv3d(channels * 3, kernel_size=1, bias=False)
        
    def forward(self, x):
        x = self.sparse_sampler(x)
        B, C, H, W, Z = x.shape
        q, k, v = (
            self.qkv(x)
            .view(B, self.num_heads, -1, H * W * Z)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        attn = (q.transpose(-2, -1) @ k).softmax(-1)
        x = (v @ attn.transpose(-2, -1)).view(B, -1, H, W, Z)
        return x
    
    
    
# def DiffAttn(X, W_q, W_k, W_v, l):
#     Q1, Q2 = split(X @ W_q)
#     K1, K2 = split(X @ W_k)
#     V = X @ W_v

#     s = 1/sqrt(d)
#     A1 = Q1@K1.transpose(-1, -2)*s
#     A2 = Q2@K2.transpose(-1, -2)*s

#     return (softmax(A1)-l*softmax(A2))@V

# def MultiHead(X, W_q, W_k, W_v, W_o, l):
#     O = GroupNorm([DiffAttn(X, W_qi, W_ki, W_vi, l) for i in range(h)])
#     O = O*(1-l_init)
#     return concat(O)@W_o
    
    
    
# class GlobalSparseMamba(nn.Module):
#     def __init__(self, channels, r, heads):
#         super(GlobalSparseMamba, self).__init__()
#         self.head_dim = channels // heads
#         self.num_heads = heads
        
#         self.sparse_sampler = nn.AvgPool3d(kernel_size=1, stride=r)
#         # qkv
#         self.qkv = nn.Conv3d(channels, channels * 3, kernel_size=1, bias=False)
#         self.back = nn.Conv3d(channels*3, channels, kernel_size=1, bias=False)
#         self.mamba = Mamba(
#             d_model=channels * 3,  # d_model을 qkv 출력 크기와 일치시킴
#             d_state=16,
#             d_conv=4,
#             expand=2,
#         ).to("cuda")
    
#     def forward(self, x):
#         x = self.sparse_sampler(x)
#         B, C, H, W, Z = x.shape
        
#         # BLD 차원 설정 (B, L, D)
#         BLD = self.qkv(x).view(B, C * 3, H * W * Z)
        
#         # Mamba 모듈에 맞게 차원을 변환
#         x = self.mamba(BLD.transpose(1, 2)).transpose(1, 2)
        
#         # 원래 차원으로 재구성
#         x = self.back(x.view(B, -1, H, W, Z))
        
#         return x


class LocalReverseDiffusion(nn.Module):
    def __init__(self, channels, r, lazy=False):
        super(LocalReverseDiffusion, self).__init__()
        self.norm = nn.GroupNorm(num_groups=1, num_channels=channels)
        self.conv_trans = nn.ConvTranspose3d(
            channels, channels, kernel_size=r, stride=r, groups=channels
        ) if not lazy else nn.LazyConvTranspose3d(channels, kernel_size=r, stride=r, groups=channels)
        self.pointwise_conv = nn.Conv3d(channels, channels, kernel_size=1, bias=False) if not lazy else nn.LazyConv3d(channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv_trans(x)
        x = self.norm(x)
        x = self.pointwise_conv(x)
        return x


class Block(nn.Module):
    def __init__(self, channels, r, heads, lazy=False):
        super(Block, self).__init__()

        self.patch1 = PatchPartition(channels, lazy=lazy)
        self.LocalRC = LocalRepresentationsCongregation(channels, lazy=lazy)
        self.LineConv1 = LineConv(channels, lazy=lazy)
        self.patch2 = PatchPartition(channels, lazy=lazy)
        self.GlobalST = GlobalSparseTransformer(channels, r, heads, lazy=lazy)
        self.LocalRD = LocalReverseDiffusion(channels, r, lazy=lazy)
        self.LineConv2 = LineConv(channels, lazy=lazy)

    def forward(self, x):
        x = self.patch1(x) + x
        x = self.LocalRC(x) + x
        x = self.LineConv1(x) + x
        x = self.patch2(x) + x
        x = self.LocalRD(self.GlobalST(x)) + x
        x = self.LineConv2(x) + x
        return x
