import torch
from monai.metrics import CumulativeIterationMetric
from monai.metrics.utils import do_metric_reduction
from monai.utils import MetricReduction
from monai.config import TensorOrList
from typing import Union, Tuple, Any, List, Optional
from collections.abc import Sequence


class BoxCollisionDiceMetric(CumulativeIterationMetric):
    def __init__(
        self,
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
        get_not_nans: bool = False,
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.get_not_nans = get_not_nans
    
    def _compute_list(
        self, y_pred: TensorOrList, y: Optional[TensorOrList] = None, **kwargs: Any
    ) -> 'Union[torch.Tensor, List[Union[torch.Tensor, Sequence[torch.Tensor]]]]':
        """
        Execute the metric computation for `y_pred` and `y` in a list of "channel-first" tensors.

        The return value is a "batch-first" tensor, or a list of "batch-first" tensors.
        When it's a list of tensors, each item in the list can represent a specific type of metric values.

        For example, `self._compute_tensor` may be implemented as returning a list of `batch_size` items,
        where each item is a tuple of three values `tp`, `fp`, `fn` for true positives, false positives,
        and false negatives respectively. This function will return a list of three items,
        (`tp_batched`, `fp_batched`, `fn_batched`), where each item is a `batch_size`-length tensor.

        Note: subclass may enhance the operation to have multi-thread support.
        """
        if y is not None:
            ret = [
                self._compute_tensor(p.detach().unsqueeze(0), y_.detach().unsqueeze(0), **{**kwargs, 'batch_index':i})
                for i, (p, y_) in enumerate(zip(y_pred, y))
            ]
        else:
            ret = [self._compute_tensor(p_.detach().unsqueeze(0), None, **{**kwargs, 'batch_index':i}) for i, p_ in enumerate(y_pred)]

        # concat the list of results (e.g. a batch of evaluation scores)
        if isinstance(ret[0], torch.Tensor):
            return torch.cat(ret, dim=0)  # type: ignore[arg-type]
        # the result is a list of sequence of tensors (e.g. a batch of multi-class results)
        if isinstance(ret[0], (list, tuple)) and all(isinstance(i, torch.Tensor) for i in ret[0]):
            return [torch.cat(batch_i, dim=0) for batch_i in zip(*ret)]
        return ret
    
    def compute_loss(self,y_pred, y, box1, box2, smooth_nr = 1e-6, smooth_dr=1e-9):
        #dice = (2*iou)/(a+b)
        #box1 : det
        #box2 : gt
        try:
            iou_box, interArea = self.get_iouBox(box1,box2)
            
            if interArea > 0:
                boxA = torch.clamp(iou_box - box1[:3].repeat(1,2),min=0).squeeze(0)
                boxB = torch.clamp(iou_box - box2[:3].repeat(1,2), min=0).squeeze(0)
                
                inter = y_pred[:,boxA[0]:boxA[3],boxA[1]:boxA[4],boxA[2]:boxA[5]]*y[:,boxB[0]:boxB[3],boxB[1]:boxB[4],boxB[2]:boxB[5]]
                if len(inter.shape)==3:
                    inter = inter.unsqueeze(0)
                        
                inter_sum = torch.sum(inter,dim=(1,2,3))
                    
                pred_sum = torch.sum(y_pred,dim=(1,2,3))
                target_sum = torch.sum(y, dim=(1,2,3))    

                score = (2*inter_sum+smooth_nr)/((pred_sum)+(target_sum)+smooth_dr)

            else:
                score=torch.zeros(y_pred.shape[0], device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            
            if torch.any(score>1 or score<0):
                raise
            
            return score
        except Exception as err:
            print("y_pred shape : ",y_pred.shape)
            print("y shape : ",y.shape)
            print("box1 : ",box1,", ", box1.shape)
            print("box2 : ",box2,", ", box2.shape)
            print("iou_box : ", iou_box, ", ", iou_box.shape)
            print("interArea : ", interArea)
            print("boxA : ",boxA,", ", boxA.shape)
            print("boxB : ",boxB,", ", boxB.shape)
            print("inter_sum : ", inter_sum)
            print("pred_sum : ", pred_sum)
            print("target_sum : ", target_sum)
            print("target unique : ", torch.unique(y))
            print("pred unique : ", torch.unique(y_pred))
            
            raise err
    
    def get_iouBox(self,box1, box2):
        xA = torch.max(box1[0], box2[0])
        yA = torch.max(box1[1], box2[1])
        zA = torch.max(box1[2], box2[2])
        xB = torch.min(box1[3], box2[3])
        yB = torch.min(box1[4], box2[4])
        zB = torch.min(box1[5], box2[5])
        
        interArea = torch.clamp(xB - xA, min=0) * torch.clamp(yB - yA, min=0) * torch.clamp(zB - zA, min=0)

        iou_box = torch.tensor([xA, yA, zA, xB, yB, zB], device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
          
        return iou_box, interArea.item()
        
    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor, det_bbox: torch.Tensor, gt_bbox: torch.Tensor, batch_index: int) -> torch.Tensor:
        """
        Overridden method to compute the metric for each batch.

        Args:
            y_pred (torch.Tensor): Predicted bounding boxes.
            y (torch.Tensor): Ground truth bounding boxes.

        Returns:
            torch.Tensor: Computed dice values for the batch.
        """
        
        return self.compute_loss(y_pred=y_pred[0], y=y[0], box1=det_bbox[batch_index].squeeze((0,1)), box2=gt_bbox[batch_index].squeeze((0,1)))
    
    def aggregate(
        self, reduction: Union[MetricReduction, str, None] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Overridden method to aggregate the metrics over all batches.

        Args:
            reduction (MetricReduction | str | None): Reduction type to apply to the aggregated data.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: The aggregated metric.
        """
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("The data to aggregate must be a PyTorch Tensor.")

        # Apply metric reduction
        f, not_nans = do_metric_reduction(data, reduction or self.reduction)
        return (f, not_nans) if self.get_not_nans else f
