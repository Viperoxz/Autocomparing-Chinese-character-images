import torch
import torch.nn as nn
import torchvision.models as models

class ResNetSiameseNetwork(nn.Module):
    def __init__(self):
        super(ResNetSiameseNetwork, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        # Thay đổi input channel từ 3 thành 1 (grayscale)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Lấy đặc trưng từ tầng trước fully connected (512 chiều)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])  # Loại bỏ fc cuối
        self.fc = nn.Sequential(
            nn.Linear(512 * 2 * 2, 256),  # Giảm từ 512*2*2 (output ResNet18 64x64) xuống 256
            nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3)
        )

    def forward_once(self, x):
        # x: [batch_size, 1, 64, 64]
        output = self.resnet(x)  # [batch_size, 512, 2, 2]
        output = output.view(output.size(0), -1)  # Flatten: [batch_size, 512*2*2]
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2