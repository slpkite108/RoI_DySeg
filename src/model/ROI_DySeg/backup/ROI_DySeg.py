import torch
import torch.nn as nn
from datetime import datetime

from .Encoder import Encoder
from .Decoder import Decoder
#from src.visModel import VisModel
from src.model.registry import register_model

@register_model('ROI_DySeg')
class ROI_DySeg(nn.Module):
    def __init__(
        self,
        in_channels=4,
        out_channels=3,
        embed_dim=96,
        embedding_dim=64,
        channels=(24, 48, 60),
        blocks=(1, 2, 3, 2),
        heads=(1, 2, 4, 4),
        r=(4, 2, 2, 1),
        conv_r=(4, 4, 4, 4),
        dropout=0.3,
    ):
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            embed_dim: deepest semantic channels
            embedding_dim: position code length
            channels: selection list of downsampling feature channel
            blocks: depth list of slim blocks
            heads: multiple set list of attention computations in parallel
            r: list of stride rate
            dropout: dropout rate
        Examples::
            # for 3D single channel input with size (128, 128, 128), 3-channel output.
            >>> net = SlimUNETR(in_channels=4, out_channels=3, embedding_dim=64)

            # for 3D single channel input with size (96, 96, 96), 2-channel output.
            >>> net = SlimUNETR(in_channels=1, out_channels=2, embedding_dim=27)

        """
        
        super(ROI_DySeg, self).__init__()
        self.Encoder = Encoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            embedding_dim=embedding_dim,
            channels=channels,
            blocks=blocks,
            heads=heads,
            r=r,
            conv_r=conv_r,
            dropout=dropout,
        )
        self.Decoder = Decoder(
            out_channels=out_channels,
            embed_dim=embed_dim,
            channels=channels,
            blocks=blocks,
            heads=heads,
            r=r,
            conv_r=conv_r,
            dropout=dropout,
        )

    def forward(self, x):
        if x.shape[-1] == 16:
            depth = 2
        elif x.shape[-1] == 32:
            depth = 1
        elif x.shape[-1] == 64:
            depth = 0
        else:
            print("error")
            print(x.shape)
            print(x.shape[-1])
            raise
        
        embeding, hidden_states_out, x_shape = self.Encoder(x, depth)
        x = self.Decoder(embeding, hidden_states_out, x_shape, depth)

        
        #exit(0)
        return x
    
    # def visualize(self, batch = 4, channel = 1, w=64, h=64, d=64, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), logger=None,  *args, **kwargs):
    #     x = torch.zeros(size=(batch, channel, w, h, d)).to(device)
    #     VisModel(self, x, device, logger, *args, **kwargs)
    #     pass
