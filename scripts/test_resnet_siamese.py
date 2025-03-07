import torch
import cv2
import os
import numpy as np
from models.resnet_siamese.resnet_siamese_model import ResNetSiameseNetwork
from utils.loss import ContrastiveLoss

def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot load image at {image_path}")
    img = img.astype(np.float32) / 255.0  # Normailze 0-1
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, 64, 64]
    return img_tensor

def test_resnet_siamese(image1_path, image2_path, checkpoint_path, threshold, device):
    model = ResNetSiameseNetwork().to(device)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    img1 = load_image(image1_path).to(device)
    img2 = load_image(image2_path).to(device)

    with torch.no_grad():
        output1, output2 = model(img1, img2)
        diff = torch.sqrt(torch.sum((output1 - output2) ** 2, dim=1))
        predicted = (diff <= threshold).float()
        prediction = "Similar" if predicted.item() == 1 else "Dissimilar"
        print(f"Distance: {diff.item():.4f}, Prediction: {prediction}")
    return prediction

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = 'models/resnet_siamese/best_resnet_model.pth'
    threshold = 0.4  # Thay bằng threshold tối ưu từ best_params

    test_samples_dir = 'data/test_samples'
    for i in range(1, 3):  # Giả sử có 2 cặp test: sample1, sample2
        img1_path = os.path.join(test_samples_dir, f'sample{i}_img1.png')
        img2_path = os.path.join(test_samples_dir, f'sample{i}_img2.png')
        print(f"\nTesting pair {i}: {img1_path} vs {img2_path}")
        test_resnet_siamese(img1_path, img2_path, checkpoint_path, threshold, device)