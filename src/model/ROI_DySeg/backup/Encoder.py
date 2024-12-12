import torch
import torch.nn as nn

from .Block import Block

class Stage(nn.Module):
    def __init__(self, dim_in, dim_out, block, r, conv_r, heads, idx=None, lazy=False, use_restore=True):
        super(Stage, self).__init__()
        #self.DWconv = DepthwiseConvLayer(dim_in=dim_in, dim_out=dim_out, r=conv_r, lazy=lazy)
        # if use_restore:
        #     self.RestoreConv = TransposedConvLayer(dim_in=dim_out, dim_out=dim_out, r=conv_r//2 if not conv_r==1 else 1, lazy=lazy)
        # self.use_restore = use_restore
        blocks = []
        for _ in range(block):
            blocks.append(Block(channels=dim_out, r=r, conv_r=conv_r, heads=heads, lazy=lazy))
        self.blocks = nn.Sequential(*blocks)
    
    def forward(self, x):
        #print(x.shape)
        #x = self.DWconv(x)
        #print(x.shape)
        x = self.blocks(x)
        # if self.use_restore:
        #     x = self.RestoreConv(x)
        #print(x.shape)
        return x

class DepthwiseConvLayer(nn.Module):
    def __init__(self, dim_in, dim_out, r, lazy=False):
        super(DepthwiseConvLayer, self).__init__()
        self.depth_wise = nn.Conv3d(dim_in, dim_out, kernel_size=r, stride=r) if not lazy else nn.LazyConv3d(dim_out,kernel_size=r,stride=r)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim_out)

    def forward(self, x):
        x = self.depth_wise(x)
        print(x.shape)
        x = self.norm(x)
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

class Encoder(nn.Module):
    def __init__(
        self,
        in_channels=4,
        embed_dim=384,
        embedding_dim=27,
        channels=(48, 96, 240),
        blocks=(1, 2, 3, 2),
        heads=(1, 2, 4, 8),
        r=(4, 2, 2, 1),
        conv_r=(4,2,2,1),
        dropout=0.3,
    ):
        super(Encoder, self).__init__()
        Stages = []
        for i in range(0, 4): 
            if i == 0:
                Stages.append(Stage(dim_in=in_channels, dim_out=channels[i], block=blocks[i], r=r[i],conv_r=conv_r[i], heads=heads[i], idx = i, lazy=True))
            elif i == 3:
                Stages.append(Stage(dim_in=channels[i-1], dim_out=embed_dim, block=blocks[i], r=r[i],conv_r=conv_r[i], heads=heads[i], idx = i, use_restore=False))
            else:
                Stages.append(Stage(dim_in=channels[i-1], dim_out=channels[i], block=blocks[i], r=r[i],conv_r=conv_r[i], heads=heads[i], idx = i, lazy=True))
        self.Stages = nn.ModuleList(Stages)
        
        self.In_Conv_List = nn.ModuleDict(
            {
                '1':DepthwiseConvLayer(dim_in=in_channels, dim_out=channels[1], r=1, lazy=True), #depth=1 32
                '2':DepthwiseConvLayer(dim_in=in_channels, dim_out=channels[2], r=2, lazy=True) #depth=2 16
            }
            # [
            #     nn.Conv3d(in_channels, channels[0], kernel_size=1, stride=1),
            #     nn.Conv3d(in_channels, channels[1], kernel_size=1, stride=1)
            # ]
        )
        
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, embedding_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
        
        self.check = [64, 32, 16]

    def forward(self, x, depth=0): #depth = 0:64, 1:32, 2:16
        hidden_states_out = []
        
        if depth == 1 or depth == 2:
            x = self.In_Conv_List[str(depth)](x)
            #print(x.shape)
            print(x.shape)
        for idx in range(depth, 3): #0, 1, 2
            #assert self.check[idx] == x.shape[-1]
            print("encoder: ",idx)
            x = self.Stages[idx](x)
            hidden_states_out.append(x)
            
        x = self.Stages[3](x)
        B, C, W, H, Z = x.shape
        #print(x.shape)
        x = x.flatten(2).transpose(-1, -2)
        #print(x.shape)
        x = x + self.position_embeddings
        x = self.dropout(x)
       
        #print(x.shape)
        return x, hidden_states_out, (B, C, W, H, Z)
