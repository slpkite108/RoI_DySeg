import numpy as np
from monai.transforms import Pad as pad_transform
from typing import Tuple, List
import torch

class AddPadding:
    """
    입력 이미지에 지정된 bbox와 target 크기에 따라 패딩을 추가하는 변환입니다.
    """

    def __init__(self, target: Tuple[int, int, int], mode: str = "constant", **kwargs):
        """
        Args:
            target: 패딩 후 목표 크기 (depth, height, width)입니다.
            mode: 패딩 모드입니다. 'constant', 'edge', 'reflect', 'wrap' 중 하나를 사용할 수 있습니다.
        """
        self.target = np.array(target)
        self.mode = mode
        self.kwargs = kwargs

    def __call__(self, image, bbox: List[List[int]]):
        """
        Args:
            image: 패딩할 입력 이미지입니다. (C, D, H, W) 형식
            bbox: 각 채널에 대한 패딩 시작점과 끝점을 나타내는 좌표 리스트 (shape: [C, 6])입니다.
                  예: [[d_start, h_start, w_start, d_end, h_end, w_end], ...]
        """
        padded_channels = []
        
        # 각 채널에 대해 padding을 개별적으로 적용
        for c in range(image.shape[0]):
            bbox_start = np.array(bbox[c][:3])  # Start coordinates for this channel
            bbox_end = np.array(bbox[c][3:])    # End coordinates for this channel
            
            # 현재 이미지의 크기 (depth, height, width)
            current_shape = np.array(image.shape[-3:])

            # 필요한 padding 계산
            pad_before = np.maximum(0, bbox_start)      # bbox 시작점까지의 거리
            pad_after = np.maximum(0, self.target - bbox_end)  # bbox 끝점부터 target 크기까지의 거리

            # padding 인자 형식에 맞게 [(before_d, after_d), (before_h, after_h), (before_w, after_w)]
            padding = [(int(pad_before[i]), int(pad_after[i])) for i in range(3)]

            # Pad를 사용하여 패딩 적용
            padded_image = pad_transform(padding=padding, mode=self.mode)(image[c].unsqueeze(0))
            
            # 패딩된 채널 이미지 추가
            padded_channels.append(padded_image.squeeze(0))

        # 각 채널을 다시 결합
        padded_image = torch.stack(padded_channels, dim=0)

        return padded_image

# 예제 사용법
# transform = AddPadding(target=(150, 150, 150), mode="constant")

# # img와 bbox 예시
# img = torch.rand((2, 128, 128, 128))  # C, D, H, W 순서
# bbox = [[20, 20, 20, 100, 100, 100], [30, 30, 30, 110, 110, 110]]  # 각 채널에 대한 (d_start, h_start, w_start, d_end, h_end, w_end)

# # transform 적용
# padded_img = transform(img, bbox)
# print(padded_img.shape)  # 예상 출력: torch.Size([2, 150, 150, 150]) 또는 설정된 target 크기에 맞게 출력
