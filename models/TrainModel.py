from ClipBertViewPredictor import ClipBertViewPredictor
from DatasetCreator import Dataset
import torch
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
from tqdm import tqdm 
import matplotlib.pyplot as plt

def custom_msle_loss(y_true, y_pred): # to change after for efficiency
    y_true = torch.exp(y_true) - 1
    y_pred = torch.exp(y_pred) - 1
    return torch.mean((torch.log1p(y_true) - torch.log1p(y_pred))**2)

def train_model(model, train_dataset, val_dataset, epochs=20, lr=1e-4, batch_size=32):

    training_losses = []
    validation_losses = []

    print("Training model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    # Custom collate function
    def collate_fn(batch):
        return {
            "image": torch.stack([x["image"] for x in batch]),
            "title": [x["title"] for x in batch],
            "description": [x["description"] for x in batch],
            "tabular": torch.stack([x["tabular"] for x in batch]),
            "target": torch.stack([x["target"] for x in batch])
        }

    print("Creating DataLoader...")
    train_loader = DataLoader(train_dataset.data, batch_size=batch_size, collate_fn=collate_fn)
    criterion = custom_msle_loss
    optimizer = optim.Adam(model.parameters(), lr=lr)

    val_loader = DataLoader(val_dataset.data, batch_size=batch_size, collate_fn=collate_fn)

    for epoch in range(epochs):
        print(f"Starting epoch {epoch+1}/{epochs}")
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch in progress_bar:
            # Move to device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                      for k, v in batch.items() if k != "target"}
            targets = batch["target"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs).squeeze() # Why ?
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        training_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Training Loss: {avg_loss:.4f}")

        # Validation

        print("Validating model...")
        model.eval()
        val_loss = 0.0
        all_targets = []
        all_predictions = []

        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validating", leave=False)
            for batch in progress_bar:
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                          for k, v in batch.items() if k != "target"}
                targets = batch["target"].to(device)

                outputs = model(inputs).squeeze()
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(outputs.cpu().numpy())
            
                progress_bar.set_postfix(loss=loss.item())
            
        avg_val_loss = val_loss / len(val_loader)
        validation_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1}/{epochs} - Validation Loss: {avg_val_loss:.4f}")
    
    return training_losses, validation_losses




def test_model(model, dataset, batch_size=32):

    losses = []

    print("Testing model...")
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

    loader = DataLoader(dataset.data, batch_size=batch_size, collate_fn=collate_fn)
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Validating")
        for batch in progress_bar:
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                      for k, v in batch.items() if k != "target"}
            targets = batch["target"].to(device)

            outputs = model(inputs).squeeze()

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(outputs.cpu().numpy())

    test_msle = custom_msle_loss(all_targets, all_predictions)
    print(f"Test MSLE: {test_msle:.4f}")
    losses.append(test_msle)
    return {
        "msle": test_msle,
        "predictions": all_predictions,
        "targets": all_targets
    }, losses

if __name__ == '__main__':
    # Train and test
    training_data = Dataset()
    val_data = Dataset("./dataset/processed_validation_set.csv")
    model = ClipBertViewPredictor()
    train_losses, val_losses = train_model(model, training_data, val_data)
    # Save the model
    torch.save(model.state_dict(), './models/first_model.pth')
    #dic, test_losses = test_model(model, val_data)
    #print(dic["msle"])

    plt.plot(range(len(train_losses)), train_losses, label='Train Loss', color='blue')
    plt.plot(range(len(val_losses)), val_losses, label='Val Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss (MSLE)')
    plt.legend()
    plt.savefig('./artifacts/loss_plot.png')