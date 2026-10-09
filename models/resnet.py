import torch
from torch import nn
from torchvision.models import resnet18


class RespiratoryResNet18(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.network = resnet18(weights=None)

        self.network.conv1 = nn.Conv2d(
            1,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        self.network.fc = nn.Linear(
            512,
            num_classes
        )


    def forward(self, feature):


        feature = feature.unsqueeze(1)

        output = self.network(feature)

        return output
