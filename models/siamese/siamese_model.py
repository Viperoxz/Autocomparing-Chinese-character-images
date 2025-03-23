import torch
import torch.nn as nn

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding='same'),
            nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            nn.ReLU(), nn.BatchNorm2d(128), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding='same'),
            nn.ReLU(), nn.BatchNorm2d(256), nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, padding='same'),
            nn.ReLU(), nn.BatchNorm2d(512), nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, kernel_size=3, padding='same'),
            nn.ReLU(), nn.BatchNorm2d(512), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512, bias=False),
            nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.3),
            nn.Linear(512, 256)
        )

    def forward_once(self, x):
        output = self.cnn(x)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2