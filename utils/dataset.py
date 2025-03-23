import os
import cv2
import numpy as np
import h5py
import random
import torch
from torch.utils.data import Dataset

class SiameseDataset(Dataset):
    def __init__(self, h5_file, train_mode=False):
        if not os.path.exists(h5_file):
            raise FileNotFoundError(f"HDF5 file not found at {h5_file}")
        with h5py.File(h5_file, 'r') as f:
            print(f"Loading {h5_file} into RAM...")
            start_time = time.time()
            self.img1 = np.array(f['img1'], dtype=np.float32)
            self.img2 = np.array(f['img2'], dtype=np.float32)
            self.labels = np.array(f['labels'], dtype=np.float32)
            print(f"img1 shape: {self.img1.shape}, min/max: {self.img1.min()}/{self.img1.max()}")
            print(f"img2 shape: {self.img2.shape}, min/max: {self.img2.min()}/{self.img2.max()}")
            self.train_mode = train_mode
            print(f"Loaded in {time.time() - start_time:.2f}s")

    def augment_image(self, img):
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        if random.random() < 0.7:
            k = random.randint(0, 3)
            img_tensor = torch.rot90(img_tensor, k, [1, 2])
        if random.random() < 0.7:
            noise = torch.normal(0, 0.1, img_tensor.shape)
            img_tensor = torch.clamp(img_tensor + noise, 0, 1)
        if random.random() < 0.5:
            shift = random.uniform(-0.1, 0.1)
            img_tensor = torch.roll(img_tensor, shifts=int(shift * 64), dims=1)
            img_tensor = torch.roll(img_tensor, shifts=int(shift * 64), dims=2)
        if random.random() < 0.5:
            brightness = random.uniform(0.8, 1.2)
            img_tensor = torch.clamp(img_tensor * brightness, 0, 1)
        return img_tensor.squeeze(0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img1 = self.img1[idx]
        img2 = self.img2[idx]
        if self.train_mode:
            img1 = self.augment_image(img1)
            img2 = self.augment_image(img2)
        else:
            img1 = torch.tensor(img1, dtype=torch.float32)
            img2 = torch.tensor(img2, dtype=torch.float32)
        return img1.unsqueeze(0), img2.unsqueeze(0), torch.tensor(self.labels[idx], dtype=torch.float32)