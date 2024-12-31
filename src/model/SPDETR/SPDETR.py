# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from src.utils.misc import (NestedTensor, nested_tensor_from_tensor_list)

from .backbone import build_backbone
from .transformer import build_transformer

from src.model.registry import register_model

from .position_encoding import build_position_encoding

@register_model('SPDETR')
class SPDETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, args):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = args.num_queries
        self.transformer = build_transformer(args)
        hidden_dim = self.transformer.d_model
        num_classes = args.num_classes
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(args.num_queries, hidden_dim)
        self.backbone = build_backbone(args)
        self.position_encoding = build_position_encoding(args)
        
        # proj = []
        # for i in range(4):
        #     proj.append(nn.Conv2d(self.backbone.num_channels//(2**(3-i)), hidden_dim, kernel_size=1))
        # self.input_proj = nn.ModuleList(proj)
        
        s_proj = []
        for i in range(4):
            s_proj.append(nn.Conv2d(3*(4*(2**i))*(4*(2**i)), hidden_dim, kernel_size=1))
        self.src_proj = nn.ModuleList(s_proj)
        
        self.input_proj = nn.Conv2d(self.backbone.num_channels, hidden_dim, kernel_size=1)
        self.aux_loss = args.aux_loss
        
    def split_sample(self, sample, p_length = (32,32)):
        #sample.shape [b, 3, H, W]
        batch_size, channels, H, W = sample.shape
        
        assert H % p_length[0] == 0 and W % p_length[1] == 0, "H and W must be divisible by patch_size"
        patch_size = (H//p_length[0],W//p_length[1])
        
        patches = sample.unfold(2, patch_size[0], patch_size[1]).unfold(3, patch_size[0], patch_size[1])
        
        # patches shape: [batch_size, channels, num_patches_H, num_patches_W, patch_size, patch_size]
        num_patches_H = patches.size(2)
        num_patches_W = patches.size(3)
        num_patches = num_patches_H * num_patches_W
        # Rearrange to [batch_size, num_patches, channels, patch_size, patch_size]
        patches = patches.contiguous().view(batch_size, channels * patch_size[0]* patch_size[1], num_patches_H, num_patches_W)
        patches = patches.permute(0, 1, 2, 3)
        
        return patches
    
    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        #sources = []
        if isinstance(samples, (list, torch.Tensor)):
            #for i in range(4):
                #sources.append(self.split_sample(samples,p_length=(32//(2**i),32//(2**i)))) 
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        
        feat, mask = features[0].decompose()
        #pos = pos[0]
        feat = self.input_proj(feat)
        mask = mask.flatten(1)
        
        # for lvl, src in enumerate(sources):
        #     sources[lvl] = self.src_proj[lvl](src)
        
        
        assert mask is not None
        # for src in src_list:
        #     print(src.shape)
        hs = self.transformer(feat, mask, self.query_embed.weight)[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        # print(outputs_class.shape)
        # print(outputs_coord.shape)
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    