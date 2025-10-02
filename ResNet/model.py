import torch
from torch import nn
from torchsummary import summary

class Residual(nn.Module):
    def __init__(self, input_channels, output_channels, downsample = False):
        super(Residual, self).__init__()
        if downsample == True:
            self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=1, stride=2, padding=0)
            stride = 2
        else:
            self.conv1 = None
            stride = 1

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, stride=stride, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(output_channels),
            nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_channels),
        )
        self.relu = nn.ReLU()
    def forward(self, x):
        y = self.model(x)
        if self.conv1 is not None:
            x = self.conv1(x)
        out = y + x
        return self.relu(out)

class ResNet18(nn.Module):
    def __init__(self, Residual, input_channels, num_classes):
        super(ResNet18, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
        )
        self.block2 = nn.Sequential(
            Residual(64, 64),
            Residual(64, 64)
        )
        self.block3 = nn.Sequential(
            Residual(64, 128, True),
            Residual(128, 128),
            Residual(128, 256, True),
            Residual(256, 256),
            Residual(256, 512, True),
            Residual(512, 512)
        )
        self.block4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode ="fan_out", nonlinearity= "relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mymodel = ResNet18(Residual, 1, 2).to(device)
    print(summary(mymodel, (1, 28, 28)))