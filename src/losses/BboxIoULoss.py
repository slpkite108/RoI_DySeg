

class iouLoss(nn.Module):
    def __init__(self,):
        super(iouLoss, self).__init__()
        
    def forward(self, preds, targets):
        iou = self.compute_iou(preds, targets)
        loss = 1- iou
        return loss.mean()
        
            
    def compute_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """
        Computes Intersection over Union (IoU) between two bounding boxes.

        Args:
            box1 (torch.Tensor): Predicted bounding boxes, shape (N, 6), where N is the number of boxes. [depth_min, height_min, width_min, depth_max, height_max, width_max]
            box2 (torch.Tensor): Ground truth bounding boxes, shape (N, 6).

        Returns:
            torch.Tensor: IoU scores per bounding box, shape (N,).
        """
           
        xA = torch.max(box1[:, :, 0], box2[:, :, 0])
        yA = torch.max(box1[:, :, 1], box2[:, :, 1])
        zA = torch.max(box1[:, :, 2], box2[:, :, 2])
        xB = torch.min(box1[:, :, 3], box2[:, :, 3])
        yB = torch.min(box1[:, :, 4], box2[:, :, 4])
        zB = torch.min(box1[:, :, 5], box2[:, :, 5])

        interArea = torch.clamp(xB - xA, min=0) * torch.clamp(yB - yA, min=0) * torch.clamp(zB - zA, min=0)

        # Calculate union
        box1Area = (box1[:, :, 3] - box1[:, :, 0]) * (box1[:, :, 4] - box1[:, :, 1]) * (box1[:, :, 5] - box1[:, :, 2])
        box2Area = (box2[:, :, 3] - box2[:, :, 0]) * (box2[:, :, 4] - box2[:, :, 1]) * (box2[:, :, 5] - box2[:, :, 2])
        unionArea = box1Area + box2Area - interArea
        # Compute IoU
        iou = interArea/unionArea

        return iou