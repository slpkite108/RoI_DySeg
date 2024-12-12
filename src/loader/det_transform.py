# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""
import random
from monai import transforms
import torch
import torchvision.transforms.functional as F

from src.utils.box_ops import box_xyxy_to_cxcywh
from src.utils.misc import interpolate


class FlipBbox:
    """
    Custom transform to flip bounding boxes horizontally or vertically.
    Assumes bbox format is [x_min, y_min, x_max, y_max].
    """

    def __init__(self, spatial_axis, image_size):
        self.spatial_axis = spatial_axis
        self.image_size = image_size  # Tuple (width, height)

    def __call__(self, bbox):
        if self.spatial_axis == 0:
            # Vertical flip
            bbox[:, [1, 3]] = self.image_size[1] - bbox[:, [3, 1]]
        elif self.spatial_axis == 1:
            # Horizontal flip
            bbox[:, [0, 2]] = self.image_size[0] - bbox[:, [2, 0]]
        return bbox


def adjust_bbox_after_crop(bbox, crop_size, crop_start):
    """
    Adjusts bounding boxes after cropping.
    """
    # Crop start coordinates
    x_start, y_start = crop_start
    # Adjust bbox coordinates by subtracting the crop start
    bbox[:, [0, 2]] -= x_start
    bbox[:, [1, 3]] -= y_start
    # Clip bbox to be within the crop size
    bbox[:, 0] = bbox[:, 0].clamp(0, crop_size[0])
    bbox[:, 1] = bbox[:, 1].clamp(0, crop_size[1])
    bbox[:, 2] = bbox[:, 2].clamp(0, crop_size[0])
    bbox[:, 3] = bbox[:, 3].clamp(0, crop_size[1])
    return bbox


def adjust_bbox_after_resize(bbox, original_size, new_size):
    """
    Adjusts bounding boxes after resizing.
    """
    if isinstance(original_size, int):
        original_size = [original_size, original_size]
    if isinstance(new_size, int):
        new_size = [new_size, new_size]
    scale_x = new_size[0] / original_size[0]
    scale_y = new_size[1] / original_size[1]
    bbox[:, [0, 2]] = bbox[:, [0, 2]] * scale_x
    bbox[:, [1, 3]] = bbox[:, [1, 3]] * scale_y
    return bbox


def det_transform(configs, mode):

    if mode == 'train':
        transform = transforms.Compose([
            transforms.LoadImaged(keys=["x", "y", "z"], ensure_channel_first=True, image_only=False),
            # Ensure 'slice' has channel first
            # Normalize 'slice'
            transforms.NormalizeIntensityd(keys=["x", "y", "z"], nonzero=True, channel_wise=True),
            transforms.RandAdjustContrastd(keys=["x", "y", "z"], prob=0.5, gamma=(0.7, 1.5)),
            transforms.RandGaussianNoised(keys=["x", "y", "z"], prob=0.3, mean=0.0, std=0.05),
            transforms.RandGaussianSmoothd(keys=["x", "y", "z"], prob=0.3, sigma_x=(0.5,1.5), sigma_y=(0.5,1.5), sigma_z=(0.5,1.5)),
            #transforms.RepeatChanneld(keys=["x", "y", "z"], repeats=3),  # 채널을 3개로 복제 (1, D, H, W -> 3, D, H, W)
        ])
    else:
        transform = transforms.Compose([
            transforms.LoadImaged(keys=["x", "y", "z"], ensure_channel_first=True, image_only=False),
            # Ensure 'slice' has channel first
            # Normalize 'slice'

            transforms.NormalizeIntensityd(keys=["x", "y", "z"], nonzero=True, channel_wise=True),
            #transforms.RepeatChanneld(keys=["x", "y", "z"], repeats=3),  # 채널을 3개로 복제 (1, D, H, W -> 3, D, H, W)
        ])

    return transform
