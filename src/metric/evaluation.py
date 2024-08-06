import torch

def dice_coefficient(prediction, target, threshold=0.5):
    """
    Dice 계수를 계산합니다.
    Args:
        prediction (torch.Tensor): 모델의 예측 값. 값의 범위는 [0, 1].
        target (torch.Tensor): 실제 마스크.
        threshold (float): 이진 분류를 위한 임계값.
    Returns:
        float: Dice 계수.
    """
    prediction = (prediction > threshold).float()
    target = (target > threshold).float()
    intersection = (prediction * target).sum()
    dice = (2. * intersection + 1e-6) / (prediction.sum() + target.sum() + 1e-6)
    return dice

def calculate_precision_recall(prediction, target, threshold=0.5):
    """
    정밀도와 재현율을 계산합니다.
    Args:
        prediction (torch.Tensor): 모델의 예측 값.
        target (torch.Tensor): 실제 마스크.
        threshold (float): 이진 분류를 위한 임계값.
    Returns:
        tuple: (정밀도, 재현율)
    """
    prediction = (prediction > threshold).float()
    target = (target > threshold).float()
    true_positive = (prediction * target).sum()
    predicted_positive = prediction.sum()
    actual_positive = target.sum()

    precision = true_positive / (predicted_positive + 1e-6)
    recall = true_positive / (actual_positive + 1e-6)
    return precision.item(), recall.item()

def calculate_accuracy(prediction, target, threshold=0.5):
    """
    정확도를 계산합니다.
    Args:
        prediction (torch.Tensor): 모델의 예측 값.
        target (torch.Tensor): 실제 마스크.
        threshold (float): 이진 분류를 위한 임계값.
    Returns:
        float: 정확도.
    """
    prediction = (prediction > threshold).float()
    target = (target > threshold).float()
    correct = (prediction == target).sum()
    total = target.numel()
    accuracy = correct.float() / total
    return accuracy.item()
