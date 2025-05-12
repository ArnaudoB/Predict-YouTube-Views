from ClipBertViewPredictor import ClipBertViewPredictor
from DatasetCreator import Dataset
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import numpy as np
from tqdm import tqdm  # Added tqdm import

def train_model(model, dataset, epochs=10, lr=1e-4, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Custom collate function
    def collate_fn(batch):
        return {
            "image": torch.cat([x["image"] for x in batch]),
            "title": [x["title"] for x in batch],
            "description": [x["description"] for x in batch],
            "tabular": torch.stack([x["tabular"] for x in batch]),
            "target": torch.stack([x["target"] for x in batch])
        }

    loader = DataLoader(dataset.data, batch_size=batch_size, collate_fn=collate_fn)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch in progress_bar:
            # Move to device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                      for k, v in batch.items() if k != "target"}
            targets = batch["target"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch["target"])
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

def msle_loss(y_true, y_pred):
    y_true = np.exp(np.array(y_true))
    y_pred = np.exp(np.array(y_pred))
    return np.mean((np.log(y_true+1) - np.log(y_pred+1))**2)

def test_model(model, dataset, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Custom collate function
    def collate_fn(batch):
        return {
            "image": torch.stack([x["image"] for x in batch]),  # Use stack instead of cat
            "title": [x["title"] for x in batch],
            "description": [x["description"] for x in batch],
            "tabular": torch.stack([x["tabular"] for x in batch]),
            "target": torch.stack([x["target"] for x in batch])
        }

    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                      for k, v in batch.items() if k != "target"}
            targets = batch["target"].to(device)

            outputs = model(inputs).squeeze()

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(outputs.cpu().numpy())

    test_msle = msle_loss(all_targets, all_predictions)
    return {
        "msle": test_msle,
        "predictions": all_predictions,
        "targets": all_targets
    }

# Train and test
training_data = Dataset()
val_data = Dataset("validation_set.csv")
model = ClipBertViewPredictor()
train_model(model, training_data)
print(test_model(model, val_data))
