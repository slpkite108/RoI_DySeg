# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2, penalize_fn=False, fn_weight=1.0, iou_threshold=0.5):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)
    
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]
    
    giou = iou - (area - union) / area
    
    if penalize_fn:
        # For each ground truth box, find the predicted box with maximum IoU
        max_iou_per_gt, max_iou_idx = iou.max(dim=0)  # [M], [M]

        # Identify ground truth boxes that have no sufficient match
        fn_mask = max_iou_per_gt < iou_threshold  # [M]

        # Calculate false negative areas for each unmatched ground truth box
        if fn_mask.any():
            # Select the ground truth boxes that are considered false negatives
            fn_gt_boxes = boxes2[fn_mask]  # [num_fn, 4]

            # For these boxes, compute the area not covered by any predicted box
            # Compute the union of all predicted boxes
            # To simplify, we can compute the total area covered by predicted boxes intersecting the FN ground truth boxes

            # Compute intersection areas between FN ground truth boxes and all predicted boxes
            inter_lt_fn = torch.max(fn_gt_boxes[:, None, :2], boxes1[:, :2])  # [num_fn, N, 2]
            inter_rb_fn = torch.min(fn_gt_boxes[:, None, 2:], boxes1[:, None, 2:])  # [num_fn, N, 2]
            inter_wh_fn = (inter_rb_fn - inter_lt_fn).clamp(min=0)  # [num_fn, N, 2]
            inter_area_fn = inter_wh_fn[:, :, 0] * inter_wh_fn[:, :, 1]  # [num_fn, N]

            # Compute the union of all intersections for each FN ground truth box
            # Assuming predicted boxes may overlap, we need to compute the total covered area without double-counting overlaps
            # However, computing exact union area for multiple overlapping boxes is complex
            # As an approximation, sum the intersection areas
            # Note: This approximation may overcount the covered area if predicted boxes overlap

            covered_area = inter_area_fn.sum(dim=1)  # [num_fn]

            # Compute the false negative area as the ground truth area minus the covered area
            gt_area = ((fn_gt_boxes[:, 2] - fn_gt_boxes[:, 0]) *
                        (fn_gt_boxes[:, 3] - fn_gt_boxes[:, 1]))  # [num_fn]
            fn_areas = gt_area - covered_area  # [num_fn]
            fn_areas = fn_areas.clamp(min=0)  # Ensure non-negative

            # Total false negative area
            total_fn_area = fn_areas.sum()

            # Normalize the penalty by the total ground truth area to keep it scale-invariant
            total_gt_area = ((boxes2[:, 2] - boxes2[:, 0]) *
                                (boxes2[:, 3] - boxes2[:, 1])).sum()

            if total_gt_area > 0:
                fn_penalty = fn_weight * (total_fn_area / total_gt_area)
            else:
                fn_penalty = torch.tensor(0.0, device=boxes1.device)

        giou = giou - fn_penalty
            # Optionally, you can subtract the penalty from giou or return it separately
            # Here, we'll return it separately for better flexibility
            # giou = giou - fn_penalty

    return giou


def pad_bboxes(bboxes, padding, target_size = (128,128)):
    """
    여러 개의 bbox에 패딩을 추가하고, 타겟 사이즈 내에 있도록 조정합니다.

    Parameters:
    - bboxes (torch.Tensor): [N, 4] 형태의 텐서, 각 bbox는 [x1, y1, x2, y2] 형식
    - padding (int 또는 float): 각 방향으로 추가할 패딩의 크기
    - target_size (list 또는 tuple): [width, height] 형태의 타겟 사이즈

    Returns:
    - torch.Tensor: 패딩이 적용되고 타겟 사이즈 내로 조정된 [N, 4] 형태의 텐서
    """
    if not isinstance(bboxes, torch.Tensor):
        raise TypeError(f"bboxes는 torch.Tensor 타입이어야 합니다.\n bbox: {bboxes}")
    if bboxes.dim() != 2 or bboxes.size(1) != 4:
        if bboxes.dim() == 1 and bboxes.size(0) == 4:
            bboxes = bboxes.unsqueeze(0)
        else:
            raise ValueError(f"bboxes는 [N, 4] 형태의 텐서여야 합니다.\n bbox shape: {bboxes.shape}")
    if not isinstance(target_size, (list, tuple)) or len(target_size) != 2:
        raise ValueError("target_size는 [width, height] 형태의 리스트나 튜플이어야 합니다.")
    
    target_width, target_height = target_size

    # 패딩 적용
    padded_bboxes = bboxes.clone()
    padded_bboxes[:, 0] = padded_bboxes[:, 0] - padding  # x1
    padded_bboxes[:, 1] = padded_bboxes[:, 1] - padding  # y1
    padded_bboxes[:, 2] = padded_bboxes[:, 2] + padding  # x2
    padded_bboxes[:, 3] = padded_bboxes[:, 3] + padding  # y2

    # 타겟 사이즈 내로 조정
    padded_bboxes[:, 0] = padded_bboxes[:, 0].clamp(min=0)
    padded_bboxes[:, 1] = padded_bboxes[:, 1].clamp(min=0)
    padded_bboxes[:, 2] = padded_bboxes[:, 2].clamp(max=target_width)
    padded_bboxes[:, 3] = padded_bboxes[:, 3].clamp(max=target_height)

    return padded_bboxes