import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv3dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv3dBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

class FCOS3DHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FCOS3DHead, self).__init__()
        # Classifier head
        self.cls_head = nn.Sequential(
            Conv3dBlock(in_channels, in_channels, 3, 1, 1),
            nn.Conv3d(in_channels, num_classes, 1)
        )
        # Box regression head
        self.box_head = nn.Sequential(
            Conv3dBlock(in_channels, in_channels, 3, 1, 1),
            nn.Conv3d(in_channels, 4, 1)  # x, y, z, depth
        )
    
    def forward(self, x):
        cls_logits = self.cls_head(x)
        box_regression = self.box_head(x)
        return cls_logits, box_regression

class FCOS3D(nn.Module):
    def __init__(self, backbone, num_classes):
        super(FCOS3D, self).__init__()
        self.backbone = backbone
        self.fcos_head = FCOS3DHead(backbone.out_channels, num_classes)

    def forward(self, images):
        features = self.backbone(images)
        cls_logits, box_regression = self.fcos_head(features)
        return cls_logits, box_regression

# 예시: 간단한 백본 모델
class SimpleBackbone3D(nn.Module):
    def __init__(self):
        super(SimpleBackbone3D, self).__init__()
        self.layer1 = Conv3dBlock(1, 64)  # 예시로, 입력 채널을 1로 설정
        self.out_channels = 64  # FCOS3DHead에 전달될 채널 수

    def forward(self, x):
        return self.layer1(x)

# 모델 사용 예
# backbone = SimpleBackbone3D()
# model = FCOS3D(backbone, num_classes=2)  # 배경 포함 클래스 수
