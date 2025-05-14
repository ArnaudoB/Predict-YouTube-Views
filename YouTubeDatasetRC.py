from torch.utils.data import Dataset
from PIL import Image
import torch
import pandas as pd
import numpy as np
from torchvision import transforms

class YouTubeDataset(Dataset):
    def __init__(self, csv_path="./dataset/processed_training_set.csv", root_dir="./dataset/train_val/", ratio=1.0, training = True):
        self.df = pd.read_csv(csv_path, sep=";")
        self.df = self.df.sample(frac=ratio, random_state=42)
        self.root_dir = root_dir
        self.training = training
        self.tabular_columns = [col for col in self.df.columns if col not in ['title', 'description', 'id', 'logviews']]

        self.transform = transforms.ToTensor()
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = f"{self.root_dir}/{row['id']}.jpg"
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        if self.training:
            sample = {
                "image": image,
                "title": row["title"],
                "description": row["description"],
                "tabular": torch.tensor(row[self.tabular_columns].values.astype(np.float32)),
                "target": torch.tensor(row["logviews"], dtype=torch.float32)
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