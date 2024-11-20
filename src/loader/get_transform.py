import torch
import monai
import numpy as np
from typing import Dict, Hashable, Mapping, Sequence
from monai import transforms
from easydict import EasyDict
from torchvision.transforms import v2

def get_transform(configs, mode):
    spatial_size = (configs.run.loader.dataset.scale,configs.run.loader.dataset.scale,configs.run.loader.dataset.scale)

    if mode == 'train':
        transform = [
            transforms.LoadImaged(keys=["image", "label"], image_only=False),
            transforms.EnsureChannelFirstd(keys=["image","label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes='RAS'),
            transforms.EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelClassesd(keys=["label"], target_label=configs.run.loader.target_labels),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
        
        if configs.run.loader.dataset.scale != configs.run.loader.dataset.image_scale:
            transform.extend([    
                BBoxSpatialCropd(keys=["image","label"], bbox_key="bbox", prefix=["", ""]),
                transforms.Resized(keys=["image","label"], spatial_size=spatial_size,mode=("bilinear","nearest"),),
                
                transforms.RandCoarseDropoutd(keys=["image"], prob=0.2, holes=100, spatial_size = int(configs.run.loader.dataset.scale/10), fill_value=0),
                
                transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                
                transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
                ])
            ####
        else:
            transform.extend([
                transforms.RandCoarseDropoutd(keys=["image"], prob=0.2, holes=100, spatial_size = int(configs.run.loader.dataset.scale/10), fill_value=0),
                
                transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                
                transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ])
    else:
        transform = [
            #Printd(keys=['bbox'], datakeys=['bbox']),
            transforms.LoadImaged(keys=["image", "label"], image_only=False),
            transforms.EnsureChannelFirstd(keys=["image","label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes='RAS'),
            transforms.EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelClassesd(keys=["label"], target_label=configs.run.loader.target_labels),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ]
            
        
        if configs.run.loader.dataset.scale != configs.run.loader.dataset.image_scale:
            if mode in ['generation']:
                transform.extend([
                    #Printd(keys=['bbox','image_meta_dict'], datakeys=['bbox']),
                    BBoxSpatialCropd(keys=["image"], bbox_key="bbox", prefix=["gt_"]),
                    transforms.Resized(keys=["image"], spatial_size=spatial_size,mode=("bilinear"),),
                    ])
            elif mode in ['inference']:
                transform.extend([
                    #Printd(keys=['bbox','image_meta_dict'], datakeys=['bbox']),
                    BBoxSpatialCropd(keys=["image","label"], bbox_key="bbox", prefix=["","gt_"]),
                    transforms.Resized(keys=["image","label"], spatial_size=spatial_size,mode=("bilinear","nearest"),),
                    ])
            else:
                transform.extend([
                    #Printd(keys=['bbox','image_meta_dict'], datakeys=['bbox']),
                    BBoxSpatialCropd(keys=["image","label"], bbox_key="bbox", prefix=["", ""]),
                    transforms.Resized(keys=["image","label"], spatial_size=spatial_size,mode=("bilinear","nearest"),),
                    ])
        
        transform = transforms.Compose(transform)
    
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

class BBoxSpatialCropd(transforms.MapTransform, transforms.Randomizable):
    """
    각 채널별 bbox 좌표를 사용하여 이미지를 크롭하는 커스텀 변환입니다.
    """

    def __init__(
        self,
        keys: Sequence[Hashable],
        bbox_key: Hashable,
        allow_missing_keys: bool = False,
        prefix: Sequence[str] = None
    ):
        """
        Args:
            keys: 변환할 대상의 키 목록입니다.
            bbox_key: bbox 좌표가 저장된 키입니다.
            allow_missing_keys: True로 설정하면 키가 없어도 예외를 발생시키지 않습니다.
            prefix: 원본 이미지를 저장할 새로운 키의 접두사 목록입니다. 각 키에 대해 개별 접두사를 설정합니다.
        """
        super().__init__(keys, allow_missing_keys)
        self.bbox_key = bbox_key
        self.prefix = prefix or [""] * len(keys)  # 기본적으로 빈 문자열로 설정

        if len(self.prefix) != len(keys):
            raise ValueError("`prefix`와 `keys`의 길이는 같아야 합니다.")

    def __call__(self, data: Mapping[Hashable, any]) -> Dict[Hashable, any]:
        d = dict(data)
        bboxes = d[self.bbox_key]  # bboxes는 (c, 6) 형태로 각 채널에 대한 bbox 좌표를 포함

        for key, pref in zip(self.keys, self.prefix):
            if key in d:
                image = d[key]  # c, w, h, d 형태의 4D 이미지

                # 원본 이미지를 새로운 키로 저장
                if pref:
                    d[f"{pref}{key}"] = image

                cropped_channels = []
                for c in range(image.shape[0]): # c,w,h,d
                    bbox = bboxes[c]  # 채널에 맞는 bbox 좌표
                    roi_start = [bbox[0], bbox[1], bbox[2]]
                    roi_end = [bbox[3], bbox[4], bbox[5]]
                    #print(bbox)#10 31 90 50 77 116
                    # 각 채널을 crop하여 저장
                    #print(image[c].shape)
                    cropped_channel = transforms.SpatialCrop(roi_start=roi_start, roi_end=roi_end)(image[c].unsqueeze(0))
                    cropped_channels.append(cropped_channel.squeeze(0))

                # 각 채널을 crop한 후 기존 키에 결합하여 저장 추후 resize까지 포함
                d[key] = np.stack(cropped_channels, axis=0)

        return d



class Printd:
    def __init__(self, keys, datakeys):
        self.keys = keys
        self.datakeys = datakeys

    def __call__(self, data):
        print(f'target keys: {self.keys}')
        print(f'datatype: {type(data)}')
        print(f'data.keys(): {data.keys()}')
        for key in self.keys:
            obj = data[key]
            print('print ',key," : ",obj)
            
        for key in self.datakeys:
            image = data[key]
            #print(image.array[:,:, 0:4,1:4, 2:4])
            # 이미지의 현재 크기를 기반으로 최대 길이 계산
            print('unique: ',torch.unique(image))
            print('sum: ',image.sum())
            print(key," : ",image.shape)
            print(key, " : ", image.device)
        
        return data