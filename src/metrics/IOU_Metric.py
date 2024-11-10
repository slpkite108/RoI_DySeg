import torch
import torch.nn as nn

class IOU_Metric(nn.Module):
    def __init__(self, smooth=1e-6, threshold=0.5):
        """
        IOU Metric 초기화 함수
        Args:
            smooth (float): 계산에서 0으로 나누는 문제를 방지하기 위한 작은 값
            threshold (float): 확률을 이진화할 임계값
        """
        super(IOU_Metric, self).__init__()
        self.smooth = float(smooth)
        self.threshold = threshold
    
    def forward(self, inputs, targets, **kwargs):
        """
        IOU 계수 계산
        Args:
            inputs (Tensor): 모델의 출력 예측값 (확률 형태, [batch, C, D, H, W])
            targets (Tensor): 타겟 레이블 (이진 형태, [batch, C, D, H, W])
        Returns:
            Tensor: 배치 내 각 샘플의 IOU 값의 평균
        """
        with torch.no_grad():  # 그레이디언트 비활성화
            # 입력이 확률 형태일 경우 이진화
            inputs = (inputs > self.threshold).float()
            
            # 배치 차원을 유지하면서 평탄화
            batch_size = inputs.size(0)
            inputs = inputs.view(batch_size, -1)
            targets = targets.view(batch_size, -1)
            
            # 교집합 및 합집합 계산
            intersection = (inputs * targets).sum(dim=1)
            union = inputs.sum(dim=1) + targets.sum(dim=1) - intersection
            
            # IOU 점수 계산
            iou_score = (intersection + self.smooth) / (union + self.smooth)
            
            # 배치 내 평균 IOU 반환
            return iou_score.mean()
