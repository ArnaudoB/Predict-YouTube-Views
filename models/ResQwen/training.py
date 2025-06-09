import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from models.ResQwen.ResQwen import ResQwen
from models.ResQwen.ResQwenDataset import ResQwenDataset
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch import optim
from tqdm import tqdm 
import matplotlib.pyplot as plt
import torch
import wandb
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def custom_msle_loss(y_true, y_pred): 
    """
    Custom loss function for the challenge's metric
    """
    err = (y_true - y_pred)**2
    weights = torch.ones_like(err)
    weights[(y_true > 10)] = 3.0
    weighted_err = err * weights
    return torch.mean(weighted_err)



def train_model(model, lr, train_dataset=ResQwenDataset(ratio=1.0, csv_path="./dataset/final_training_set.csv", root_dir="./dataset/train_val/"),
                 val_dataset=ResQwenDataset("./dataset/validation_set.csv", "./dataset/train_val/", ratio=1.0),
                 custom_test=ResQwenDataset("./dataset/test_set.csv", "./dataset/train_val/", ratio=1.0),
                 batch_size=128, 
                 optimizer_name="Adam",
                 early_stopping=True, 
                 patience=5, 
                 phases=[4, 2],
                 criterion1=torch.nn.MSELoss(),
                 criterion2=torch.nn.HuberLoss(delta=1.0, reduction='mean'),
                 use_wandb=True,
                 wandb_project_name="youtube-views-prediction",
                 wandb_run_name=None,
                 plot_each_epoch=False):
    """
    Training function for the ResQwen model.
    Args:
    model: The ResQwen model to train.
    lr : list with learning rates for different parameter groups (low lr for finetuning and 'high lr' for the rest).
    train_dataset: The training dataset.
    val_dataset: The validation dataset.
    custom_test: The custom test dataset.
    batch_size: The batch size for training.
    optimizer: The optimizer to use (default is Adam, we can maybe try other optimizers but the code is not ready for that).
    early_stopping: Whether to use early stopping (default is False).
    patience: Number of epochs with no improvement after which training will be stopped (default is 5).
    phases: List of epochs for each phase of the training (default is [20, 10]). The first phase is for the MLPs in the model, the second phase is for the ResNet finetuning.
    criterion1: The first loss function to use (default is MSLE). It is used for the training.
    criterion2: The second loss function to use (default is HuberLoss). It is used for information purposes.
    use_wandb: Whether to use wandb for logging (default is True).
    wandb_project_name: The name of the wandb project (default is "youtube-views-prediction").
    wandb_run_name: The name of the wandb run (default is None, which will use the default run name).
    plot_each_epoch: Whether to plot the predictions and histograms for each epoch (default is False).

    Returns:
    training_losses: List of training losses for each epoch.
    validation_losses: List of validation losses for each epoch.
    """

    name1 = criterion1.__class__.__name__ if hasattr(criterion1, '__class__') else criterion1.__name__
    name2 = criterion2.__class__.__name__ if hasattr(criterion2, '__class__') else criterion2.__name__

    if use_wandb:
        run = wandb.init(project=wandb_project_name, name=wandb_run_name, config={
            "model_type": model.__class__.__name__,
            "learning_rates": lr,
            "batch_size": batch_size,
            "optimizer": optimizer_name,
            "early_stopping": early_stopping,
            "patience": patience,
            "phases": phases,
            "criterion1": name1,
            "criterion2": name2,
            "dropout": model.dropout if hasattr(model, "dropout") else "N/A",
            "freeze_qwen": model.freeze_qwen if hasattr(model, "freeze_qwen") else "N/A",
            "freeze_resnet": model.freeze_resnet if hasattr(model, "freeze_resnet") else "N/A"
        })

        wandb.config.update({
        "num_trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "dataset_size": len(train_dataset),
        "val_dataset_size": len(val_dataset),
        "architecture": str(model)
        })

        # wandb.watch(model, log="all")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    training_losses_1 = []
    training_losses_2 = []

    validation_losses_1 = []
    validation_losses_2 = []
    

    print("Beginning model training...")
    print(f"There are {len(phases)} phases in the training.")


    # Custom collate function
    def collate_fn(batch):
        return {
            "image": torch.stack([x["image"] for x in batch]),
            "title": [x["title"] for x in batch],
            "date": torch.stack([x["date"] for x in batch]),
            "channel": torch.stack([x["channel"] for x in batch]),
            "target": torch.stack([x["target"] for x in batch])
        }

    print("Creating DataLoaders...")

    # weights = train_dataset.get_weights()
    # sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=2, pin_memory=True, shuffle=True)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=2, pin_memory=True, sampler=sampler)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=2, shuffle=False, pin_memory=True)
    if custom_test is not None:
        custom_test_loader = DataLoader(custom_test, batch_size=batch_size, collate_fn=collate_fn, num_workers=2, shuffle=False, pin_memory=True)

    global_step = 0 # for wandb logging

    for i, phase in enumerate(phases):

        print(f"Phase {i+1}/{len(phases)}")

        if i == 1:

            for name, param in model.resnet.named_parameters():
                if 'resnet.7' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        elif i == 2:

            for name, param in model.qwen.model.named_parameters():
                if param.dtype not in [torch.float, torch.float16, torch.bfloat16, torch.complex64, torch.complex128]:
                    continue  # skip non-trainable dtypes
                if any(f"layers.{i}" in name for i in range(model.qwen.model.config.num_hidden_layers - model.qwen.layers_to_unfreeze, model.qwen.model.config.num_hidden_layers)):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Trainable parameters:", num_trainable)
            
        
        # Set up parameter groups with different learning rates. Based on our wandb analysis, we can set different learning rates for the different parts of the model.

        low_lr_params = []
        medium_lr_params = []
        high_lr_params = []

        for name, param in model.named_parameters():
            if param.requires_grad:  # Only include trainable parameters
                if 'resnet.7' in name:
                    low_lr_params.append(param)
                    print(f"Param with fine-tuning lr: {name}")
                elif 'date_mlp' in name or 'channel_mlp' in name or 'title_mlp' in name or 'resnet.projection' in name or 'qwen.title' in name:
                    high_lr_params.append(param)
                    print(f"Param with high lr: {name}")
                else:
                    medium_lr_params.append(param)
                    print(f"Other param with base lr: {name}")

        if optimizer_name=='Adam':

            optimizer = optim.Adam([
                {'params': low_lr_params, 'lr': lr[2]},  # Fine-tuning lr for encoders
                {'params': medium_lr_params, 'lr': lr[1]},  
                {'params': high_lr_params, 'lr': lr[0]}  # For the layers that struggle to learn
            ], weight_decay=1e-4)

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, min_lr=[lr[2]/10, lr[1]/100, lr[0]/100])

        best_val_loss = float('inf')
        counter = 0 # Early stopping counter


        for epoch in range(phase):

            print(f"Starting epoch {epoch+1}/{phase}")
            model.train()
            total_loss1 = 0.0
            total_loss2 = 0.0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{phase}", leave=False)

            all_targets = []
            all_predictions = []

            for batch in progress_bar:

                global_step += 1

                # Move to device
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items() if k != "target"} # We keep the image on CPU to save memory
                
                targets = batch["target"].to(device)

                optimizer.zero_grad()
                outputs = model(inputs).view(-1)
                
                loss1 = criterion1(outputs, targets)
                loss2 = criterion2(outputs, targets)
                if torch.isnan(loss1) or torch.isinf(loss1):
                    print("Loss is NaN or inf! Breaking...")
                    break
                loss1.backward()

                clip_grad_norm_(high_lr_params, 3.0)
                clip_grad_norm_(medium_lr_params, 1.0)
                clip_grad_norm_(low_lr_params, 0.2)

                for name, param in model.named_parameters():
                    if param.grad is not None:
                        wandb.log({f"grad_hist/{name}": wandb.Histogram(param.grad.cpu().numpy())})

                optimizer.step()

                total_loss1 += loss1.item()
                total_loss2 += loss2.item()

                progress_bar.set_postfix(loss=loss1.item())

                all_targets.extend(targets.detach().cpu().numpy())
                all_predictions.extend(outputs.detach().cpu().numpy())


            avg_loss1 = total_loss1 / len(train_loader)
            avg_loss2 = total_loss2 / len(train_loader)
            training_losses_1.append(avg_loss1)
            training_losses_2.append(avg_loss2)

            if plot_each_epoch:

                # Plot the predictions
                plt.scatter(all_targets, all_predictions, alpha=0.5)
                plt.plot([min(all_targets), max(all_targets)], [min(all_targets), max(all_targets)], 'r--')
                plt.xlabel('True Values')
                plt.ylabel('Predicted Values')
                plt.title(f'Phase {i + 1} - Epoch {epoch+1}/{phase} - Predictions vs True Values (Training set)')
                plt.savefig(f'./artifacts/train_plots/scatt/train_predictions_phase_{i + 1}_epoch_{epoch + 1}.png')
                plt.close()

                # Plot the histogram of the predictions
                plt.hist(all_predictions, bins='auto', alpha=0.5, label='Predictions')
                plt.hist(all_targets, bins='auto', alpha=0.5, label='True Values')
                plt.xlabel('Values')
                plt.ylabel('Frequency')
                plt.title(f'Phase {i + 1} - Epoch {epoch+1}/{phase} - Histogram of Predictions and True Values (Training set)')
                plt.legend()
                plt.savefig(f'./artifacts/train_plots/hist/train_histogram__phase_{i + 1}_epoch_{epoch + 1}.png')
                plt.close()

            print(f"Epoch {epoch+1}/{phase} - Training Loss for {name1} : {avg_loss1:.4f} - Training Loss for {name2} : {avg_loss2:.4f}")

            # Validation

            print("Validating model...")

            model.eval()

            val_loss1 = 0.0
            val_loss2 = 0.0

            all_targets = []
            all_predictions = []

            with torch.no_grad():

                progress_bar = tqdm(val_loader, desc="Validating", leave=False)

                for batch in progress_bar:

                    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items() if k != "target"}
                    
                    targets = batch["target"].to(device)

                    outputs = model(inputs).view(-1)
                    loss1 = criterion1(outputs, targets)
                    loss2 = criterion2(outputs, targets)

                    val_loss1 += loss1.item()
                    val_loss2 += loss2.item()
                    all_targets.extend(targets.cpu().numpy())
                    all_predictions.extend(outputs.cpu().numpy())
                
                    progress_bar.set_postfix(loss=loss1.item())
                
            avg_val_loss1 = val_loss1 / len(val_loader)
            avg_val_loss2 = val_loss2 / len(val_loader)
            validation_losses_1.append(avg_val_loss1)
            validation_losses_2.append(avg_val_loss2)

            if use_wandb:
                wandb.log({
                    "train_loss_1": avg_loss1,
                    "train_loss_2": avg_loss2,
                    "val_loss_1": avg_val_loss1,
                    "val_loss_2": avg_val_loss2,
                    "epoch": epoch+1,
                    "phase": i+1
                }, step=global_step)

            if avg_val_loss1 < best_val_loss:

                best_val_loss = avg_val_loss1
                counter = 0

                # Save the model
                print(f"Validation loss improved to {avg_val_loss1:.4f}. Saving model...")
                torch.save(model.state_dict(), f"./models/ResQwen/ResQwen_model_phase_{i+1}.pth")

                model.qwen.model.save_pretrained(f"./models/ResQwen/qwen_ckpt/")
                model.qwen.tokenizer.save_pretrained(f"./models/ResQwen/qwen_ckpt/")

                if use_wandb:
                    # Save the model checkpoint to wandb
                    model_path = f"./models/ResQwen/ResQwen_model_phase_{i+1}.pth"
                    artifact = wandb.Artifact(f"model-checkpoint-phase{i+1}-epoch{epoch}", type="model")
                    artifact.add_file(model_path)
                    wandb.log_artifact(artifact)

            else:
                counter += 1
                if early_stopping and counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}/{phase}")
                    break
                
            scheduler.step(avg_val_loss1)

            # Display loss on unseen test set

            if custom_test is not None:
                custom_test_loss1 = 0.0
                custom_test_loss2 = 0.0

                all_custom_targets = []
                all_custom_predictions = []

                with torch.no_grad():
                    progress_bar = tqdm(custom_test_loader, desc="Testing on custom set", leave=False)

                    for batch in progress_bar:

                        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                for k, v in batch.items() if k != "target"}
                        
                        targets = batch["target"].to(device)

                        outputs = model(inputs).view(-1)
                        loss1 = criterion1(outputs, targets)
                        loss2 = criterion2(outputs, targets)

                        custom_test_loss1 += loss1.item()
                        custom_test_loss2 += loss2.item()
                        all_custom_targets.extend(targets.cpu().numpy())
                        all_custom_predictions.extend(outputs.cpu().numpy())
                    
                        progress_bar.set_postfix(loss=loss1.item())

                avg_custom_test_loss1 = custom_test_loss1 / len(custom_test_loader)
                avg_custom_test_loss2 = custom_test_loss2 / len(custom_test_loader)

                print(f"Custom Test Loss for {name1} : {avg_custom_test_loss1:.4f} - Custom Test Loss for {name2} : {avg_custom_test_loss2:.4f}")
            

            if plot_each_epoch:

                # Plot the predictions
                plt.scatter(all_targets, all_predictions, alpha=0.5)
                plt.plot([min(all_targets), max(all_targets)], [min(all_targets), max(all_targets)], 'r--')
                plt.xlabel('True Values')
                plt.ylabel('Predicted Values')
                plt.title(f'Phase {i + 1} - Epoch {epoch+1}/{phase} - Predictions vs True Values (Validation set)')
                plt.savefig(f'./artifacts/val_plots/scatt/val_predictions_phase_{i + 1}_epoch_{epoch + 1}.png')
                plt.close()

                # Plot the histogram of the predictions
                plt.hist(all_predictions, bins='auto', alpha=0.5, label='Predictions')
                plt.hist(all_targets, bins='auto', alpha=0.5, label='True Values')
                plt.xlabel('Values')
                plt.ylabel('Frequency')
                plt.title(f'Phase {i + 1} - Epoch {epoch+1}/{phase} - Histogram of Predictions and True Values (Validation set)')
                plt.legend()
                plt.savefig(f'./artifacts/val_plots/hist/val_histogram_phase_{i + 1}_epoch_{epoch + 1}.png')
                plt.close()

            print(f"Epoch {epoch+1}/{phase} - Validation Loss for {name1} : {avg_val_loss1:.4f} - Validation Loss for {name2} : {avg_val_loss2:.4f}")
            print(f"LRs: {[param_group['lr'] for param_group in optimizer.param_groups]}")
            print("------------------------------------------------------------------------------")

            if use_wandb and epoch%3 == 0:
                df = pd.DataFrame({
                    "Target": all_targets,
                    "Prediction": all_predictions
                })
                fig = plt.figure(figsize=(6, 6))
                sns.scatterplot(x="Target", y="Prediction", data=df, alpha=0.3)
                plt.title("Predicted vs Target (Validation)")
                wandb.log({"val_scatter": wandb.Image(fig)})
    
        print(f"Phase {i+1}/{len(phases)} completed.")
        print("------------------------------------------------------------------------------")
    
    if use_wandb:
        plt.figure(figsize=(10, 5))
        plt.plot(training_losses_1, label='Train Loss', color='blue')
        plt.plot(validation_losses_1, label='Val Loss', color='red')
        if training_losses_2 and validation_losses_2:
            plt.plot(training_losses_2, label='Train Loss 2', color='orange')
            plt.plot(validation_losses_2, label='Val Loss 2', color='purple')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curves')
        plt.legend()
        wandb.log({"loss_curves": wandb.Image(plt)})
        
        # Finish the wandb run
        wandb.finish()

    plot_loss_curves(training_losses_1, validation_losses_1, name1, training_losses_2, validation_losses_2, name2, phases, save_path='./artifacts/ResQwen_loss_curves.png')
    return training_losses_1, validation_losses_1, training_losses_2, validation_losses_2


def plot_loss_curves(train_losses1, val_losses1, label1, train_losses2, val_losses2, label2, phases, save_path):
    """
    Plot the training and validation loss curves.
    Args:
    train_losses: List of training losses for each epoch.
    val_losses: List of validation losses for each epoch.
    phases: List of epochs for each phase of the training.
    save_path: Path to save the plot.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses1, label='Train Loss ' + label1, color='blue')
    plt.plot(val_losses1, label='Val Loss ' + label1, color='red')


    if train_losses2 is not None and val_losses2 is not None:
        plt.plot(train_losses2, label='Train Loss ' + label2, color='orange')
        plt.plot(val_losses2, label='Val Loss ' + label2, color='purple')
    # Add vertical lines for each phase
    phase_start = 0
    for i, phase in enumerate(phases):
        phase_end = phase_start + phase
        plt.axvline(x=phase_end, color='green', linestyle='--')
        phase_start = phase_end

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.savefig(save_path)