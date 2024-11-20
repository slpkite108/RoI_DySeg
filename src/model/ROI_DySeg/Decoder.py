import torch.nn as nn

from .Block import Block

class Stage(nn.Module):
    def __init__(self, dim_in, dim_out, block, r, conv_r, heads, idx, lazy=False, use_restore=True):
        super(Stage, self).__init__()
        blocks = []
        for _ in range(block):
            blocks.append(Block(channels=dim_in, r=r, heads=heads, lazy=lazy))
        self.blocks = nn.Sequential(*blocks)
        self.TSconv = TransposedConvLayer(dim_in=dim_in, dim_out=dim_out, r=conv_r, lazy=lazy) #r)#4212
        if use_restore:
            self.RestoreConv = DepthwiseConvLayer(dim_in=dim_in, dim_out=dim_in, r=conv_r//2 if not conv_r==1 else 1, lazy=lazy)
        self.use_restore = use_restore
    
    def forward(self, x):
        if self.use_restore:
            x = self.RestoreConv(x)
        x = self.blocks(x)
        x = self.TSconv(x)
        return x

class TransposedConvLayer(nn.Module):
    def __init__(self, dim_in, dim_out, r, lazy=False):
        super(TransposedConvLayer, self).__init__()
        self.transposed = nn.ConvTranspose3d(dim_in, dim_out, kernel_size=r, stride=r) if not lazy else nn.LazyConvTranspose3d(dim_out, kernel_size=r, stride=r)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim_out)

    def forward(self, x):
        x = self.transposed(x)
        x = self.norm(x)
        return x

class DepthwiseConvLayer(nn.Module):
    def __init__(self, dim_in, dim_out, r, lazy=False):
        super(DepthwiseConvLayer, self).__init__()
        self.depth_wise = nn.Conv3d(dim_in, dim_out, kernel_size=r, stride=r) if not lazy else nn.LazyConv3d(dim_out,kernel_size=r,stride=r)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim_out)

    def forward(self, x):
        x = self.depth_wise(x)
        x = self.norm(x)
        return x

class Decoder(nn.Module):
    def __init__(
        self,
        out_channels=3,
        embed_dim=384,
        channels=(48, 96, 240), #0, 1, 2
        blocks=(1, 2, 3, 2), #0, 1, 2, 3
        heads=(1, 2, 4, 8),
        r=(4, 2, 2, 1),
        conv_r=(4, 2, 2, 2),
        dropout=0.3,
    ):
        super(Decoder, self).__init__()
        Stages = []
        for i in range(0, 4): 
            if i == 0:
                Stages.append(Stage(dim_in=channels[i], dim_out=out_channels, block=blocks[i], r=r[i], conv_r=conv_r[i], heads=heads[i], idx = i, lazy=True))
            elif i == 3:
                Stages.append(Stage(dim_in=embed_dim, dim_out=channels[i-1], block=blocks[i], r=r[i], conv_r=conv_r[i], heads=heads[i], idx = i, use_restore=False))
            else:
                Stages.append(Stage(dim_in=channels[i], dim_out=channels[i-1], block=blocks[i], r=r[i], conv_r=conv_r[i], heads=heads[i], idx = i, lazy=True))
        self.Stages = nn.ModuleList(Stages)
        
        self.SegHeadList = nn.ModuleDict(
            {
                '1':TransposedConvLayer(dim_in=channels[0], dim_out=out_channels, r=1, lazy=True), #32
                '2':TransposedConvLayer(dim_in=channels[1], dim_out=out_channels, r=1, lazy=True) #16
            }
            # [
            #     nn.ConvTranspose3d(channels[0], out_channels, kernel_size=1, stride=1),
            #     nn.ConvTranspose3d(channels[1], out_channels, kernel_size=1, stride=1)
            # ]
        )

    def forward(self, x, hidden_states_out, x_shape ,depth=0):
        # print(x.shape)
        x = x.reshape(x_shape)
        x = self.Stages[3](x)
        
        for idx in range(2, depth-1, -1): #2, 1, 0
            x = x + hidden_states_out[idx-depth]
            x = self.Stages[idx](x)
            
        if depth == 1 or depth == 2:
            # print(x.shape) #8, 48, 8, 8, 8
            x = self.SegHeadList[str(depth)](x)
            # print(x.shape)
        return x
