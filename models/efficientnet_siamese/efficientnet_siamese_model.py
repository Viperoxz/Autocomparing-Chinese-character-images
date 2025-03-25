import torch
import torch.nn as nn
import torchvision.models as models

class EfficientNetSiameseNetwork(nn.Module):
    def __init__(self, freeze_base=True):
        super(EfficientNetSiameseNetwork, self).__init__()
        self.effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        pretrained_weights = self.effnet.features[0][0].weight
        self.effnet.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.effnet.features[0][0].weight = nn.Parameter(pretrained_weights.mean(dim=1, keepdim=True))
        self.backbone = nn.Sequential(*list(self.effnet.features.children()))
        if freeze_base:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256)
        )

    def forward_once(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2