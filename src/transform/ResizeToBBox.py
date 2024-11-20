from monai.transforms import Transform
import torch
import torch.nn.functional as F

class ResizeToBBox(Transform):
    def __init__(self, mode='bilinear'):
        """
        :param mode: 이미지 resize에 사용할 interpolation 모드 (default: 'bilinear')
        """
        self.mode = mode

    def __call__(self, img, bbox):
        """
        :param img: 입력 이미지 (torch.Tensor, shape: (C, D, H, W))
        :param bbox: 각 채널에 대한 bounding box 좌표 (torch.Tensor, shape: (C, 6) -> (xmin, ymin, zmin, xmax, ymax, zmax))
        :return: bbox 크기로 resize된 이미지 (각 채널별 bbox 크기에 맞춰 조정됨)
        """
        resized_channels = []
        
        # 각 채널마다 bounding box에 맞춰 리사이즈
        for c in range(img.shape[0]):
            xmin, ymin, zmin, xmax, ymax, zmax = bbox[c]
            bbox_shape = (xmax - xmin, ymax - ymin, zmax - zmin)
            
            # 각 채널 이미지를 bbox 크기로 리사이즈
            img_c_resized = F.interpolate(img[c].unsqueeze(0).unsqueeze(0), size=bbox_shape, mode=self.mode, align_corners=False)
            
            # (1, 1, D, H, W)에서 (D, H, W)로 차원 축소 후 저장
            resized_channels.append(img_c_resized.squeeze(0).squeeze(0))
        
        # 채널 차원을 다시 결합
        img_resized = torch.stack(resized_channels, dim=0)
        
        return img_resized

# 예제 사용법
# transform = ResizeToBBox(mode='trilinear')

# # img와 bbox 예시
# img = torch.rand((2, 128, 128, 128))  # C, D, H, W 순서
# bbox = torch.tensor([[20, 20, 20, 100, 100, 100], [30, 30, 30, 110, 110, 110]])  # 각 채널에 대한 (xmin, ymin, zmin, xmax, ymax, zmax)

# # transform 적용
# resized_img = transform(img, bbox)
# print(resized_img.shape)  # 예상 출력: torch.Size([2, 80, 80, 80]) 및 torch.Size([2, 80, 80, 80])과 비슷한 결과
