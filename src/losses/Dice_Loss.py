import torch
import torch.nn as nn

class Dice_Loss(nn.Module):
    def __init__(self, smooth=1e-6, threshold=0.5):
        super(Dice_Loss, self).__init__()
        self.smooth = float(smooth)
        self.threshold = threshold
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, preds, targets, **kwargs):
        
        preds = self.sigmoid(preds)
        
        # Flattening the tensors for easy computation
        preds = preds.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        
        # Calculating intersection and union
        intersection = (preds * targets).sum()
        dice = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        
        return 1 - dice  # Dice Loss는 일반적으로 1에서 유사도를 뺀 값으로 계산합니다.
