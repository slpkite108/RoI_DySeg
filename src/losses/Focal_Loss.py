import torch
import torch.nn as nn
import torch.nn.functional as F

class Focal_Loss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', threshold=0.5):
        """
        Focal Loss 초기화 함수
        Args:
            alpha (float): 클래스 불균형을 조절하는 파라미터 (1이면 균형)
            gamma (float): 잘 맞추기 쉬운 예측에 대한 손실을 감소시키는 파라미터
            reduction (str): 출력의 축소 방법 ('mean', 'sum', 또는 'none')
            threshold (float): (필요 시) 예측 확률을 이 값과 비교하여 이진화할 때 사용
        """
        super(Focal_Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.threshold = threshold
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, inputs, targets, **kwargs):
        """
        Focal Loss 계산
        Args:
            inputs (Tensor): 모델의 출력 예측값 (logits 형태, [b, c, d, h, w])
            targets (Tensor): 타겟 레이블 (클래스 인덱스, [b, d, h, w])
        Returns:
            Tensor: Focal Loss 값
        """
        inputs = self.sigmoid(inputs)
        
        # Cross Entropy Loss 계산 (reduction='none'으로 개별 손실 유지)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # [b, d, h, w]
        
        # 정답 클래스에 대한 확률 계산
        pt = torch.exp(-ce_loss)  # [b, d, h, w]
        
        # Focal Loss 계산
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss  # [b, d, h, w]
        
        # Reduction 방식에 따라 손실값 반환
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
