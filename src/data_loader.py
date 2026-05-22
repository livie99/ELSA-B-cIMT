import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class ELSA_B_Dataset(Dataset):
    def __init__(self, label_df, base_dir, img_col, label_col, transform=None, target_transform=None):
        self.label_df = label_df
        self.base_dir = base_dir
        self.img_col = img_col
        self.label_col = label_col
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        img_p = self.label_df[self.img_col].iloc[idx]
        img_path = os.path.join(self.base_dir, img_p)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = self.label_df[self.label_col].iloc[idx]
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
