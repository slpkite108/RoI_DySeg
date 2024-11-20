import torch
from torch.utils.data import DataLoader
from monai.data import SmartCacheDataset, Dataset
from .load_dataset_image import image_loader
from .get_transform import get_transform
from .det_transform import det_transform
import nibabel as nib

def get_loader(configs, mode='train'):

    assert mode in ['train', 'inference', 'validation', 'generation']
    subjects = image_loader(configs, mode)
    if configs.run.model.type in ['det']:
        transform = det_transform(configs, mode)
        
    elif configs.run.model.type in ['seg']:
        transform = get_transform(configs, mode)
        

    if mode in ['inference','generation']:
        dataset = Dataset(
            data=subjects,
            transform=transform,
        )
    else:
        dataset = SmartCacheDataset(
            data=subjects,
            transform=transform,
            cache_rate=1.0,
            shuffle=False,
        )
        dataset.start()

    batch_size = configs.run.loader.batch_size if (mode in ['train', 'validation']) else 1
    shuffle = True if mode in ['train', 'validation'] else False
    num_workers = 1 #configs.run.loader.num_workers
    pin_memory = False #configs.run.loader.pin_memory 
    drop_last = configs.run.loader.drop_last if not mode in ['generation'] else False
    
    loader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    
    
    return loader
    