import os
import time
import psutil
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.efficientnet_siamese.efficientnet_siamese_model import EfficientNetSiameseNetwork
from utils.dataset import SiameseDataset
from utils.loss import ContrastiveLoss
from utils.helpers import train_epoch, validate_epoch

def train_and_validate_lr_margin(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, patience=10):
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    checkpoint_path = f'effnet_siamese_checkpoint_lr_{optimizer.param_groups[0]["lr"]}_margin_{criterion.margin}.pth'
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, min_lr=0.000075)

    with tqdm(total=num_epochs, desc=f"Training lr={optimizer.param_groups[0]['lr']}, margin={criterion.margin}") as pbar:
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, _, _, _ = validate_epoch(model, val_loader, criterion, device, threshold=0)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            epoch_time = time.time() - epoch_start_time
            print(f"{epoch_time:.1f}s\tEpoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'val_loss': val_loss}, checkpoint_path)
                patience_counter = 0
                print(f"{epoch_time:.1f}s\tVal loss improved to {val_loss:.4f}, saving model...")
            else:
                patience_counter += 1

            scheduler.step(val_loss)
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break
            pbar.update(1)
    return checkpoint_path, min(val_losses)

if __name__ == "__main__":
    print(f"Available RAM before loading: {psutil.virtual_memory().available / 1024**3:.2f} GB")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_file = '/kaggle/input/create-trainset/train_pairs.h5'
    val_file = '/kaggle/input/create-trainset/valid_pairs.h5'

    train_dataset = SiameseDataset(train_file, train_mode=True)
    val_dataset = SiameseDataset(val_file, train_mode=False)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    lr_values = [0.0003, 0.0005, 0.0007, 0.001]
    margin_values = [1.0, 1.5]
    NUM_EPOCHS = 20
    PATIENCE = 10

    best_val_loss = float('inf')
    best_lr_margin = None
    best_checkpoint = None

    for lr in lr_values:
        for margin in margin_values:
            print(f"\n=== Trying lr={lr}, margin={margin} ===")
            model = EfficientNetSiameseNetwork(freeze_base=True).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
            criterion = ContrastiveLoss(margin=margin)
            
            checkpoint_path, val_loss = train_and_validate_lr_margin(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, device, PATIENCE)
            print(f"lr={lr}, margin={margin} - Best Val Loss={val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_lr_margin = (lr, margin)
                best_checkpoint = checkpoint_path

    print(f"\nBest Hyperparameters: lr={best_lr_margin[0]}, margin={best_lr_margin[1]}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")