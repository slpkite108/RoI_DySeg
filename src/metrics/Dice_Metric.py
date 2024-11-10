import torch
import torch.nn as nn

class Dice_Metric(nn.Module):
    def __init__(self, smooth=1e-6, threshold=0.5):
        """
        Dice Metric 초기화 함수

        Args:
            smooth (float): 계산에서 0으로 나누는 문제를 방지하기 위한 작은 값
            threshold (float): 확률을 이진화할 임계값
        """
        super(Dice_Metric, self).__init__()
        self.smooth = float(smooth)
        self.threshold = threshold

    def forward(self, inputs, targets, **kwargs):
        """
        Dice 계수 계산

        Args:
            inputs (Tensor): 모델의 출력 예측값 (확률 형태, [batch, C, D, H, W])
            targets (Tensor): 타겟 레이블 (이진 형태, [batch, C, D, H, W])

        Returns:
            Tensor: 배치 내 각 샘플의 Dice 계수 값의 평균
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
            sum_inputs = inputs.sum(dim=1)
            sum_targets = targets.sum(dim=1)
            
            # Dice 계수 계산
            dice_score = (2. * intersection + self.smooth) / (sum_inputs + sum_targets + self.smooth)

            # 배치 내 평균 Dice 계수 반환
            return dice_score.mean()
