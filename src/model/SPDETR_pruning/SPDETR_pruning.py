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

@register_model('SPDETR_pruning')
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
        
        proj = []
        for i in range(4):
            proj.append(nn.Conv2d(self.backbone.num_channels//(2**(3-i)), hidden_dim, kernel_size=1))
        self.input_proj = nn.ModuleList(proj)
        #self.input_proj = nn.Conv2d(self.backbone.num_channels, hidden_dim, kernel_size=1)
        self.aux_loss = args.aux_loss

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
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src_list = []
        mask_list = []
        for lvl, feat in enumerate(features):
            src, mask = feat.decompose()
            #print(src.shape) torch.Size([1, 64, 32, 32])torch.Size([1, 128, 16, 16])torch.Size([1, 256, 8, 8])torch.Size([1, 512, 4, 4])

            src_list.append(self.input_proj[lvl](src))    # 레벨별 src 저장
            mask_list.append(mask.flatten(1))  # 레벨별 mask 저장
        
        for idx,mask in enumerate(mask_list):
            # print('mask:',mask_list[idx].shape)
            mask_list[idx] = F.interpolate(mask_list[idx].float().unsqueeze(1), size=mask_list[idx].shape[-1]//4, mode='linear', align_corners=False).squeeze(1)
            mask_list[idx] = mask_list[idx] > 0.5
            # print(pos[idx].shape)
            # print('mask:',mask_list[idx].shape)
        
        assert mask_list[0] is not None
        # for src in src_list:
        #     print(src.shape)
        hs = self.transformer(src_list, mask_list, self.query_embed.weight)[0]

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


# class PostProcess(nn.Module):
#     """ This module converts the model's output into the format expected by the coco api"""
#     @torch.no_grad()
#     def forward(self, outputs, target_sizes):
#         """ Perform the computation
#         Parameters:
#             outputs: raw outputs of the model
#             target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
#                           For evaluation, this must be the original image size (before any data augmentation)
#                           For visualization, this should be the image size after data augment, but before padding
#         """
#         out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

#         assert len(out_logits) == len(target_sizes)
#         assert target_sizes.shape[1] == 2

#         prob = F.softmax(out_logits, -1)
#         scores, labels = prob[..., :-1].max(-1)

#         # convert to [x0, y0, x1, y1] format
#         boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
#         # and from relative [0, 1] to absolute [0, height] coordinates
#         img_h, img_w = target_sizes.unbind(1)
#         scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
#         boxes = boxes * scale_fct[:, None, :]

#         results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

#         return results


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
    

# def build(args):
#     # the `num_classes` naming here is somewhat misleading.
#     # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
#     # is the maximum id for a class in your dataset. For example,
#     # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
#     # As another example, for a dataset that has a single class with id 1,
#     # you should pass `num_classes` to be 2 (max_obj_id + 1).
#     # For more details on this, check the following discussion
#     # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
#     num_classes = 20 if args.dataset_file != 'coco' else 91
#     if args.dataset_file == "coco_panoptic":
#         # for panoptic, we just add a num_classes that is large enough to hold
#         # max_obj_id + 1, but the exact value doesn't really matter
#         num_classes = 250
#     device = torch.device(args.device)

#     backbone = build_backbone(args)

#     transformer = build_transformer(args)

#     model = DETR(
#         backbone,
#         transformer,
#         num_classes=num_classes,
#         num_queries=args.num_queries,
#         aux_loss=args.aux_loss,
#     )
#     if args.masks:
#         model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
#     matcher = build_matcher(args)
#     weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
#     weight_dict['loss_giou'] = args.giou_loss_coef
#     if args.masks:
#         weight_dict["loss_mask"] = args.mask_loss_coef
#         weight_dict["loss_dice"] = args.dice_loss_coef
#     # TODO this is a hack
#     if args.aux_loss:
#         aux_weight_dict = {}
#         for i in range(args.dec_layers - 1):
#             aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
#         weight_dict.update(aux_weight_dict)

#     losses = ['labels', 'boxes', 'cardinality']
#     if args.masks:
#         losses += ["masks"]
#     criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
#                              eos_coef=args.eos_coef, losses=losses)
#     criterion.to(device)
#     postprocessors = {'bbox': PostProcess()}
#     if args.masks:
#         postprocessors['segm'] = PostProcessSegm()
#         if args.dataset_file == "coco_panoptic":
#             is_thing_map = {i: i <= 90 for i in range(201)}
#             postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

#     return model, criterion, postprocessors