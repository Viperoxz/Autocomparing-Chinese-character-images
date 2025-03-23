import torch
import matplotlib.pyplot as plt

def plot_hist(similar_diffs, dissimilar_diffs, threshold):
    plt.figure(figsize=(10, 6))
    plt.hist(similar_diffs, bins=50, alpha=0.5, label='Similar Pairs (Label=1)', color='blue')
    plt.hist(dissimilar_diffs, bins=50, alpha=0.5, label='Dissimilar Pairs (Label=0)', color='red')
    plt.axvline(threshold, color='green', linestyle='--', label=f'Threshold={threshold:.4f}')
    plt.xlabel('Euclidean Distance')
    plt.ylabel('Frequency')
    plt.title('Distribution of Similar and Dissimilar Pairs')
    plt.legend()
    plt.show()

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for img1, img2, label in train_loader:
        img1, img2, label = img1.to(device), img2.to(device), label.to(device)
        optimizer.zero_grad()
        output1, output2 = model(img1, img2)
        loss = criterion(output1, output2, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * img1.size(0)
    return running_loss / len(train_loader.dataset)

def validate_epoch(model, val_loader, criterion, device, threshold):
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
    return avg_loss, accuracy, similar_diffs, dissimilar_diffs