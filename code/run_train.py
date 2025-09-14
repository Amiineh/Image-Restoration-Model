"""
File: run_train.py
Description: Train Image Reconstruction network (U-Net with two decoders defined in model/image_reconstruction_model.py)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import os
from datetime import datetime
import json
from model.image_restoration_model import ImageRestorationModel
from data.img_utils import ImageRestorationDataset


# Helper function to get current time as a string to save different checkpoints (used for hyper-parameter optimization)
def get_current_time():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


config = {
    "epochs": 50,
    "batch_size" : 32,
    "lr" : 0.0001,
    "save_interval": 5,
    "train_data_path" : "../train",
    "validation_data_path" : "../validation/",
    "checkpoint_path": "../checkpoints/",
    "version": get_current_time(),
    "num_workers": 4,
}

checkpoint_path = os.path.join(config["checkpoint_path"], config["version"])
# create checkpoint folder 
os.makedirs(checkpoint_path, exist_ok=True)
# save config 
config_path = os.path.join(checkpoint_path, "config.json")
with open(config_path, 'w') as f:
    json.dump(config, f, indent=4)

# setup device for training
device = torch.device("cuda" if torch.cuda.is_available() else "CPU")
print("using device: ", device)

class CustomLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.L1Loss(reduction='sum') # we'll average manually because of masking

    def forward(self, img_gt, img_pred, mask_gt, mask_pred):
        # Reconstruction Loss:
        # mask before calculating l1:
        pixels_x = img_gt * mask_gt
        pixels_x_hat = img_pred * mask_gt

        mask_size = mask_gt.sum()
        # avoid division by zero:
        if mask_size > 0:
            l_x = self.l1(pixels_x, pixels_x_hat) / mask_size
        else:
            l_x = torch.tensor(0, 0).to(device)
        
        # Mask Loss:
        l_m = nn.functional.binary_cross_entropy(mask_pred, mask_gt, reduction='mean')

        return 2 * l_x + l_m

def save_checkpoint(model, optimizer, epoch, config, loss, path):
    # Check if the 'model_best.pth' file already exists
    if os.path.exists(path):
        print(f"Removing existing best model at {path}")
        os.remove(path)  # Remove the file if it exists
        
    checkpoint = {
        'epoch': epoch + 1,  # Store next epoch number
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config  # Optionally save the config
    }
    torch.save(checkpoint, path)
    


def train():
    # load data
    print("loading data...")
    train_dataset = ImageRestorationDataset(config["train_data_path"])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    val_dataset = ImageRestorationDataset(config["validation_data_path"])
    val_loader = DataLoader(val_dataset, 25, num_workers=config["num_workers"])

    #  load model:
    print("loading model...")
    model = ImageRestorationModel().to(device)
    # define loss function and optimizer
    loss_function = CustomLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # early stopping to avoid overfitting
    best_loss = float('inf')
    epochs_no_improve = 0
    patience = 10  # Number of epochs to wait for improvement before stopping the training
    min_delta = 0.0001  # The threshold for observing an improvement

    print("training started")
    for epoch in range(config["epochs"]):
        # training
        total_loss_train = 0
        for data in train_loader:
            orig, corrupt, mask = data
            orig = orig.to(device)
            corrupt = corrupt.to(device)
            mask = mask.to(device)

            # make prediction (forward pass):
            img_pred, mask_pred = model(corrupt)

            # calculate loss:
            loss = loss_function(orig, img_pred, mask, mask_pred)

            # backpropagate 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss_train += loss.item()

        # validation
        total_loss_val = 0
        for data in val_loader:
            orig, corrupt, mask = data
            orig = orig.to(device)
            corrupt = corrupt.to(device)
            mask = mask.to(device)

            # make prediction (forward pass):
            img_pred, mask_pred = model(corrupt)

            # calculate loss:
            loss = loss_function(orig, img_pred, mask, mask_pred)
            total_loss_val += loss.item()

        avg_train_loss = total_loss_train / len(train_loader)
        avg_val_loss = total_loss_val / len(val_loader)
        print(f'Epoch {epoch + 1}, Train Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}')

        # evaluate for early stopping 
        if avg_val_loss < best_loss - min_delta:
            best_loss = avg_val_loss
            epochs_no_improve = 0
            # Save the best model
            checkpoint_path_new = os.path.join(checkpoint_path, f"model_best.pth")
            save_checkpoint(model, optimizer, epoch, config, loss.item(), checkpoint_path_new)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break

        # Save model checkpoint 
        if (epoch + 1) % config["save_interval"] == 0 or epoch == config["epochs"] - 1:
            checkpoint_path_new = os.path.join(checkpoint_path, f"model_epoch_{epoch+1}.pth")
            save_checkpoint(model, optimizer, epoch, config, loss.item(), checkpoint_path_new)
            print(f"Checkpoint saved at {checkpoint_path_new}")


if __name__ == '__main__':
    train()