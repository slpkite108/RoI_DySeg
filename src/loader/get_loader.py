import torch
from torch.utils.data import DataLoader
from monai.data import SmartCacheDataset, Dataset
from .load_dataset_image import load_dataset_image
from .get_transform import get_transform
import nibabel as nib

class Custom3DDataset(Dataset):
    def __init__(self, subject, transform=None):
        self.subject = subject
        self.transform = transform
        
    def __len__(self):
        return len(self.subject)
    
    def __getitem__(self, idx):
        sample = self.subject[idx]
        
        sample['image'] = torch.from_numpy(nib.load(sample['image']).get_fdata()).to(dtype=torch.float32).unsqueeze(0)
        sample['label'] = nib.load(sample['label']).get_fdata()
        
        # if self.transform:
        #     sample = self.transform(sample)
        
        return sample


def get_loader(configs, mode='train'):

    assert mode in ['train', 'inference', 'validation', 'generation']
    subjects = load_dataset_image(configs, mode)
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
    