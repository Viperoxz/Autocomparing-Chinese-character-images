import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from utils.dataset import SiameseDataset
from utils.loss import ContrastiveLoss
from utils.helpers import train_epoch, validate_epoch, test, plot_training_history
from models.resnet_siamese.resnet_siamese_model import ResNetSiameseNetwork

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, threshold, patience=10):
    train_losses, val_losses, val_accuracies = [], [], []
    best_val_loss = float('inf')
    patience_counter = 0
    checkpoint_path = 'models/resnet_siamese/resnet_checkpoint.pth'
    output_dir = 'outputs/resnet_siamese/plots'
    log_file = 'outputs/resnet_siamese/logs/training_log.txt'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, min_lr=0.000075)

    with open(log_file, 'w') as f:
        for epoch in tqdm(range(num_epochs), desc="Training Progress", unit="epoch"):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_accuracy = validate_epoch(model, val_loader, criterion, device, threshold, output_dir, epoch)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            log = (f"Epoch {epoch+1}/{num_epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - "
                   f"val_acc: {val_accuracy:.2f}%\n")
            print(log)
            f.write(log)

            if val_loss < best_val_loss:
                log = f"val_loss improved from {best_val_loss:.4f} to {val_loss:.4f}, saving model...\n"
                print(log)
                f.write(log)
                best_val_loss = val_loss
                torch.save({'model_state_dict': model.state_dict(), 'val_loss': val_loss, 'threshold': threshold}, checkpoint_path)
                patience_counter = 0
            else:
                patience_counter += 1
                log = f"Patience Counter: {patience_counter}/{patience}\n"
                print(log)
                f.write(log)

            scheduler.step(val_loss)
            log = f"Current LR: {optimizer.param_groups[0]['lr']:.6f}\n"
            print(log)
            f.write(log)

            if patience_counter >= patience:
                log = f"Early stopping at epoch {epoch+1}!\n"
                print(log)
                f.write(log)
                break

    plot_training_history(train_losses, val_losses, val_accuracies, output_dir)
    log = f"Loaded best model with val_loss: {best_val_loss:.4f}\n"
    print(log)
    with open(log_file, 'a') as f:
        f.write(log)
    return train_losses, val_losses, val_accuracies

def optimize_hyperparameters(train_loader, val_loader, device, num_epochs=30, patience=10):
    best_val_acc = 0.0
    best_params = {}
    lr_values = [0.0003, 0.0005, 0.001, 0.002]
    margin_values = [1.0, 1.5, 2.0]
    threshold_values = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    log_file = 'outputs/resnet_siamese/logs/training_log.txt'

    for lr, margin, threshold in [(lr, margin, threshold) for lr in lr_values for margin in margin_values for threshold in threshold_values]:
        print(f"\nTrying hyperparameters: lr={lr}, margin={margin}, threshold={threshold}")
        model = ResNetSiameseNetwork().to(device)
        criterion = ContrastiveLoss(margin=margin)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)

        train_losses, val_losses, val_accuracies = train_model(
            model, train_loader, val_loader, criterion, optimizer, num_epochs, device, threshold, patience
        )
        best_val_acc_in_run = max(val_accuracies)
        
        if best_val_acc_in_run > best_val_acc:
            best_val_acc = best_val_acc_in_run
            best_params = {'lr': lr, 'margin': margin, 'threshold': threshold}
            torch.save({'model_state_dict': model.state_dict(), 'val_loss': min(val_losses), 'val_acc': best_val_acc_in_run}, 'models/resnet_siamese/best_resnet_model.pth')
        
        log = f"Best val_acc for this run: {best_val_acc_in_run:.2f}%\n"
        print(log)
        with open(log_file, 'a') as f:
            f.write(log)

    log = f"\nBest hyperparameters: {best_params}, Best val_acc: {best_val_acc:.2f}%\n"
    print(log)
    with open(log_file, 'a') as f:
        f.write(log)
    return best_params

def final_evaluation(best_params, train_loader, val_loader, test_loader, device):
    checkpoint = torch.load('models/resnet_siamese/best_resnet_model.pth', weights_only=False)
    model = ResNetSiameseNetwork().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    criterion = ContrastiveLoss(margin=best_params['margin'])
    optimizer = optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=1e-2)
    
    combined_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
    combined_loader = DataLoader(combined_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    
    train_losses, val_losses, val_accuracies = train_model(
        model, combined_loader, val_loader, criterion, optimizer, 30, device, best_params['threshold'], patience=10
    )
    
    test_acc = test(model, test_loader, device, best_params['threshold'])
    print(f"Final test accuracy with optimized hyperparameters: {test_acc:.2f}%")
    return test_acc

# Main execution
if __name__ == "__main__":
    print(f"Available RAM before loading: {psutil.virtual_memory().available / 1024**3:.2f} GB")

    train_dataset = SiameseDataset('data/train_pairs_part1.h5', train_mode=True)
    val_dataset = SiameseDataset('data/valid_pairs_part1.h5', train_mode=False)
    test_dataset = SiameseDataset('data/test_pairs.h5', train_mode=False)

    print(f"Train dataset: {len(train_dataset)} pairs, Ratio of similar pairs: {np.mean(train_dataset.labels == 1):.2f}")
    print(f"Validation dataset: {len(val_dataset)} pairs, Ratio of similar pairs: {np.mean(val_dataset.labels == 1):.2f}")
    print(f"Test dataset: {len(test_dataset)} pairs, Ratio of similar pairs: {np.mean(test_dataset.labels == 1):.2f}")
    print(f"Available RAM after loading: {psutil.virtual_memory().available / 1024**3:.2f} GB")

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    best_params = optimize_hyperparameters(train_loader, val_loader, device, num_epochs=30, patience=10)
    final_evaluation(best_params, train_loader, val_loader, test_loader, device)