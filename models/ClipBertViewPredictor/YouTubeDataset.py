from torch.utils.data import Dataset
from PIL import Image
import torch
import pandas as pd
import numpy as np
from torchvision import transforms

class YouTubeDataset(Dataset):
    def __init__(self, csv_path="./dataset/final_training_set.csv", root_dir="./dataset/train_val/", ratio=1.0, is_test=False):
        self.df = pd.read_csv(csv_path, sep=";")
        self.df = self.df.sample(frac=ratio, random_state=42)
        self.root_dir = root_dir
        self.is_test = is_test

        self.tabular_columns = [col for col in self.df.columns if col not in ['title', 'description', 'id', 'logviews', 'views']]

        self.transform = transforms.ToTensor()
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = f"{self.root_dir}/{row['id']}.jpg"
        image = Image.open(image_path).convert("RGB")

        if not self.is_test:
            sample = {
                "image": image,
                "title": row["title"],
                "description": row["description"],
                "tabular": torch.tensor(row[self.tabular_columns].values.astype(np.float32)),
                "target": torch.tensor(row["views"], dtype=torch.float32)
            }
        else:
            sample = {
                "image": image,
                "title": row["title"],
                "description": row["description"],
                "tabular": torch.tensor(row[self.tabular_columns].values.astype(np.float32)),
                "id": row["id"]
            }

        return sample