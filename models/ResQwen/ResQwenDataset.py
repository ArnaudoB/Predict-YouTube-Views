from torch.utils.data import Dataset
from PIL import Image
import torch
import pandas as pd
import numpy as np
from torchvision.models import ResNet50_Weights

class ResQwenDataset(Dataset):
    def __init__(self, csv_path="./dataset/final_training_set.csv", root_dir="./dataset/train_val/", ratio=1.0, is_test=False):
        self.df = pd.read_csv(csv_path, sep=";")

        self.df = self.df.sample(frac=ratio, random_state=42)

        self.root_dir = root_dir
        self.is_test = is_test

        self.date_columns = ['year'] + ['sin_month', 'cos_month', 'sin_dayofmonth', 'cos_dayofmonth', 'sin_hour', 'cos_hour', 'sin_dayofweek', 'cos_dayofweek', 'sin_dayofyear', 'cos_dayofyear']
        self.channel_columns = [col for col in self.df.columns if col not in ['title', 'description', 'id', 'logviews', 'views', 'normalized_logviews'] and col not in self.date_columns]

        weights = ResNet50_Weights.IMAGENET1K_V2
        self.transform = weights.transforms()
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = f"{self.root_dir}/{row['id']}.jpg"
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        if not self.is_test:
            sample = {
                "image": image,
                "title": row["title"],
                "channel": torch.tensor(row[self.channel_columns].values.astype(np.float32)),
                "date": torch.tensor(row[self.date_columns].values.astype(np.float32)),
                "target": torch.tensor(row["logviews"], dtype=torch.float32),
            }
        else:
            sample = {
                "image": image,
                "title": row["title"],
                "channel": torch.tensor(row[self.channel_columns].values.astype(np.float32)),
                "date": torch.tensor(row[self.date_columns].values.astype(np.float32)),
                "id": row["id"]
            }

        return sample
    
    def get_weights(self):
        logviews = self.df["logviews"].values
        weights = np.ones_like(logviews, dtype=np.float32)
        weights[(logviews > 10)] = 3.0  # bigger weight for outliers
        return torch.tensor(weights, dtype=torch.float32)