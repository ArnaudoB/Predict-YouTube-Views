import torch
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
from torchvision import transforms

class Dataset():
    
    def __init__(self, set = "./dataset/processed_training_set.csv"):
        self.set = set
        self.data = self.create_dataset()
    
    def create_dataset(self):

        #_, preprocess = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
        df = pd.read_csv(self.set, sep=";")

        # load a small part of the dataset for testing :
        df = df.sample(1000)

        tabular_columns = [col for col in df.columns if col not in ['title', 'description', 'id', 'logviews']]
        
        dataset = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            image = Image.open("./dataset/train_val/"+row['id']+".jpg").convert("RGB")

            sample = {
                "image": transforms.ToTensor()(image),
                "title": row['title'],
                "description": row['description'],
                "tabular": torch.tensor(row[tabular_columns].values.astype(np.float32)),
                "target": torch.tensor(row["logviews"], dtype=torch.float32)
            }
            dataset.append(sample)
           
        return dataset

