import torch
import cv2
import os
import numpy as np
from model import EfficientNetSiameseNetwork
from utils.loss import ContrastiveLoss

def load_image(image_path):
    """Đọc và tiền xử lý ảnh từ file."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot load image at {image_path}")
    img = img.astype(np.float32) / 255.0  # Normalize 0-1
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    return img_tensor

def test_model(image1_path, image2_path, checkpoint_path, threshold, device, margin):
    """Kiểm tra mô hình trên một cặp ảnh."""
    model = EfficientNetSiameseNetwork(freeze_base=True).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
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
    print(f"Using device: {device}")

    # Cấu hình
    checkpoint_path = "effnet_siamese_checkpoint_lr_0.0005_margin_1.0.pth"  # Thay bằng checkpoint thực tế
    threshold = 0.5  # Thay bằng threshold tối ưu
    margin = 1.0  # Thay bằng margin tối ưu từ train.py
    test_samples_dir = 'data/test_samples'  # Thư mục chứa ảnh test

    # Tạo thư mục nếu chưa tồn tại (tùy chọn)
    os.makedirs(test_samples_dir, exist_ok=True)

    # Duyệt qua các cặp ảnh mẫu
    for i in range(1, 3):  # Giả sử có 2 cặp: sample1, sample2
        img1_path = os.path.join(test_samples_dir, f'sample{i}_img1.png')
        img2_path = os.path.join(test_samples_dir, f'sample{i}_img2.png')
        if os.path.exists(img1_path) and os.path.exists(img2_path):
            print(f"\nTesting pair {i}: {img1_path} vs {img2_path}")
            test_model(img1_path, img2_path, checkpoint_path, threshold, device, margin)
        else:
            print(f"Skipping pair {i}: One or both images not found.")