import torch
import matplotlib.pyplot as plt
import os

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for img1, img2, label in train_loader:
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        optimizer.zero_grad()
        output1, output2 = model(img1, img2)
        loss = criterion(output1, output2, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * img1.size(0)
    return running_loss / len(train_loader.dataset)

def plot_hist(similar_diffs, dissimilar_diffs, threshold, output_dir, epoch):
    plt.figure(figsize=(10, 6))
    plt.hist(similar_diffs, bins=50, alpha=0.5, label='Similar Pairs (Label=1)', color='blue')
    plt.hist(dissimilar_diffs, bins=50, alpha=0.5, label='Dissimilar Pairs (Label=0)', color='red')
    plt.axvline(threshold, color='green', linestyle='--', label=f'Threshold={threshold}')
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Frequency')
    plt.title('Distribution of Similar and Dissimilar Pairs')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'epoch_{epoch+1}_hist.png'))
    plt.close()

def validate_epoch(model, val_loader, criterion, device, threshold, output_dir, epoch):
    model.eval()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    similar_diffs = []
    dissimilar_diffs = []
    
    with torch.no_grad():
        for img1, img2, label in val_loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            output1, output2 = model(img1, img2)
            diff = torch.sqrt(torch.sum((output1 - output2) ** 2, dim=1))
            loss = criterion(output1, output2, label)
            running_loss += loss.item() * img1.size(0)
            
            for d, l in zip(diff.cpu().numpy(), label.cpu().numpy()):
                if l == 1:
                    similar_diffs.append(d)
                else:
                    dissimilar_diffs.append(d)
            
            predicted = (diff <= threshold).float()
            total_correct += (predicted == label).sum().item()
            total_samples += label.size(0)
    
    avg_loss = running_loss / total_samples
    accuracy = 100 * total_correct / total_samples
    plot_hist(similar_diffs, dissimilar_diffs, threshold, output_dir, epoch)
    return avg_loss, accuracy

def test(model, test_loader, device, threshold):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for img1, img2, label in test_loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            output1, output2 = model(img1, img2)
            diff = torch.sqrt(torch.sum((output1 - output2) ** 2, dim=1))
            predicted = (diff <= threshold).float()
            total_correct += (predicted == label).sum().item()
            total_samples += label.size(0)
    return 100 * total_correct / total_samples

def plot_training_history(train_losses, val_losses, val_accuracies, output_dir):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Val Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Val Accuracy', color='green', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()