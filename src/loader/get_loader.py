from torch.utils.data import DataLoader
from monai.data import SmartCacheDataset
from .load_dataset_image import load_dataset_image
from .get_transform import get_transform

def get_loader(configs, mode='train'):

    assert mode in ['train', 'inference', 'validation', 'generation']
    subjects = load_dataset_image(configs, mode)
    transform = get_transform(configs, mode)
    # if mode == 'train':
    #     dataset = SmartCacheDataset(
    #         data=subjects,
    #         transform=transform,
    #         replace_rate=0.2,
    #         cache_num=60,
    #         num_init_workers=2,
    #         num_replace_workers=2,
    #     )
    # else:
    dataset = SmartCacheDataset(
        data=subjects,
        transform=transform,
        cache_rate=1.0,
    )
    
    batch_size = configs.run.loader.batch_size if (mode in ['train', 'validation']) else 1
    shuffle = True if mode in ['train', 'validation'] else False
    num_workers = configs.run.loader.num_workers
    pin_memory = configs.run.loader.pin_memory
    drop_last = configs.run.loader.drop_last
    
    loader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    
    dataset.start()
    
    return loader
    