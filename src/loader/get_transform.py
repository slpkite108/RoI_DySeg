import torch
import monai
import numpy as np
from typing import Dict, Hashable, Mapping
from monai import transforms
from easydict import EasyDict

def get_transform(configs, mode):
    if mode == 'train':
        transform = transforms.Compose([
            transforms.LoadImaged(keys=["image", "label"], image_only=True),
            transforms.EnsureChannelFirstd(keys=["image","label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes='RAS'),
            transforms.EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelClassesd(keys=["label"], target_label=configs.run.loader.target_labels),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ####
            transforms.RandCoarseDropoutd(keys=["image"], prob=0.2, holes=100, spatial_size = int(configs.run.loader.dataset.scale/10), fill_value=0),
            
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ])
    else:
        transform = transforms.Compose([
            transforms.LoadImaged(keys=["image", "label"], image_only=False),
            transforms.EnsureChannelFirstd(keys=["image","label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes='RAS'),
            transforms.EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelClassesd(keys=["label"], target_label=configs.run.loader.target_labels),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ])
    
    return transform

class ConvertToMultiChannelClassesd(monai.transforms.MapTransform):
    """
    TC WT ET
    Dictionary-based wrapper of :py:class:`monai.transforms.ConvertToMultiChannelBasedOnBratsClasses`.
    Convert labels to multi channels based on brats18 classes:
    label 1 is the necrotic and non-enhancing tumor core
    label 2 is the peritumoral edema
    label 4 is the GD-enhancing tumor
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
    """

    backend = [monai.utils.TransformBackends.TORCH, monai.utils.TransformBackends.NUMPY]

    def __init__(
        self,
        target_label: list,
        keys: monai.config.KeysCollection,
        allow_missing_keys: bool = False,
        tolerance: float = 1e-5,  # 추가된 허용 오차
    ):
        super().__init__(keys, allow_missing_keys)

        self.target_label = [float(l) for l in target_label]  # target_label을 float으로 변환
        self.tolerance = tolerance

    def converter(self, img: monai.config.NdarrayOrTensor):
        result = []
        if self.target_label is None or len(self.target_label) == 0:
            # img와 비교할 때 항상 float 타입 사용
            result = [torch.isclose(img, torch.tensor(1.0), atol=self.tolerance)] if isinstance(img, torch.Tensor) \
                     else [np.isclose(img, 1.0, atol=self.tolerance)]
        else:
            try:
                # target_label이 float으로 변환되었으므로 float과 안전하게 비교 가능
                if isinstance(img, torch.Tensor):
                    result = [torch.isclose(img, torch.tensor(l, dtype=img.dtype), atol=self.tolerance) for l in self.target_label]
                else:
                    result = [np.isclose(img, l, atol=self.tolerance) for l in self.target_label]
            except Exception as e:
                print('src.transform.ConvertToMultiChannelClassesd Error : ', e)
        
        if not result:
            result = [img == 1.0] if isinstance(img, torch.Tensor) else [img == 1.0]
        return (
            torch.concat(result, dim=0)
            if isinstance(img, torch.Tensor)
            else np.concatenate(result, axis=0)
        )

    def __call__(
        self, data: Mapping[Hashable, monai.config.NdarrayOrTensor]
    ) -> Dict[Hashable, monai.config.NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d



class Printd:
    def __init__(self, keys, datakeys):
        self.keys = keys
        self.datakeys = datakeys

    def __call__(self, data):
        print(self.keys)
        print(type(data))
        print(data.keys())
        for key in self.keys:
            obj = data[key]
            print(key," : ",obj)
            
        for key in self.datakeys:
            image = data[key]
            #print(image.array[:,:, 0:4,1:4, 2:4])
            # 이미지의 현재 크기를 기반으로 최대 길이 계산
            print(torch.unique(image))
            print(image.sum())
            print(key," : ",image.shape)
            print(key, " : ", image.device)
        exit(0)
        return data