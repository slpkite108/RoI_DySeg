import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose

class NiiGzDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transforms=None):
        """
        Args:
            image_paths (list of str): 이미지 파일 경로 목록.
            mask_paths (list of str): 마스크 파일 경로 목록.
            transforms (callable, optional): 적용할 변환.
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        image = nib.load(image_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()
        
        if self.transforms:
            # Transform을 이미지와 마스크에 적용하는 로직 추가
            pass
        
        return image, mask

def get_data_loaders(image_paths, mask_paths, batch_size=4, transform=None):
    dataset = NiiGzDataset(image_paths, mask_paths, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
