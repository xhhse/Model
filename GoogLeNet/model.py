import torch
from torch import nn
from torchsummary import summary
import torchvision

class Inception(nn.Module):
    # c2, c3: (x, y), c1, c4: x
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(Inception, self).__init__()
        self.ReLU = nn.ReLU()
        self.p1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=c1, kernel_size=1),
            nn.ReLU()
        )
        self.p2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=c2[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=c2[0], out_channels=c2[1], kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.p3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=c3[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=c3[0], out_channels=c3[1], kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.p4 = nn.Sequential(
            nn.MaxPool2d(3, padding=1, stride=1),
            nn.Conv2d(in_channels=in_channels, out_channels=c4, kernel_size=1),
            nn.ReLU()
        )
    def forward(self, x):
        path1 = self.p1(x)
        path2 = self.p2(x)
        path3 = self.p3(x)
        path4 = self.p4(x)
        return torch.cat((path1, path2, path3, path4), dim = 1)

class GoogLeNet(nn.Module):
    def __init__(self, Inception, num_classes):
        super(GoogLeNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3,64, 7, 2, 3 ),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 192, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
        )
        self.block3 = nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32),
            Inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(3, 2, 1)
        )
        self.block4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (128, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(3, 2, 1)
        )
        self.block5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)), # 1*1*1024
            nn.Flatten(),
            nn.Linear(1024, num_classes)
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
        x = self.block5(x)
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 2
    model = GoogLeNet(Inception, num_classes).to(device)
    print(summary(model, (1, 224, 224)))

