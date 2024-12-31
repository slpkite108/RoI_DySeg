# call models
from .SlimUNETR import SlimUNETR
from .ROI_DySeg import ROI_DySeg
from .DETR import DETR
from .SPDETR import SPDETR
from .SPDETR_pruning import SPDETR_pruning
from .SPDETR_backconv import SPDETR_backconv


from .registry import getModel