import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

class ChangeDetectionDataset(Dataset):
    def __init__(self, pre_eo_dir, pre_sar_dir, post_eo_dir, post_sar_dir, mask_dir):
        self.pre_eo_dir = pre_eo_dir
        self.pre_sar_dir = pre_sar_dir
        self.post_eo_dir = post_eo_dir
        self.post_sar_dir = post_sar_dir
        self.mask_dir = mask_dir
        self.files = os.listdir(mask_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]

        pre_eo = transform(Image.open(os.path.join(self.pre_eo_dir, file)).convert('L'))
        pre_sar = transform(Image.open(os.path.join(self.pre_sar_dir, file)).convert('L'))
        post_eo = transform(Image.open(os.path.join(self.post_eo_dir, file)).convert('L'))
        post_sar = transform(Image.open(os.path.join(self.post_sar_dir, file)).convert('L'))
        mask = transform(Image.open(os.path.join(self.mask_dir, file)).convert('L'))

        pre = torch.cat([pre_eo, pre_sar], dim=0)
        post = torch.cat([post_eo, post_sar], dim=0)
        return pre, post, mask
