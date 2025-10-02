import torch
from setuptools.namespaces import flatten
from torch import nn
from torchsummary import summary
import torchvision

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 6, 5, 1, 2),
            nn.Sigmoid(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, 5, 1),
            nn.Sigmoid(),
            nn.AvgPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(400, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10)
        )
    def forward(self, input):
        output = self.model(input)
        return output


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet().to(device)
    print(summary(model, (1, 28, 28)))