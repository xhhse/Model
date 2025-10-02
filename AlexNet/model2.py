import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class AlexNet(nn.Module):
    def __init__(self, input_shape=(1, 227, 227)):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
        )

        # 🔧 自動計算展平後的大小
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            flatten_size = self.features(dummy).view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Linear(flatten_size, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = F.softmax(x, dim=1)
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNet()
    model = model.to(device)
    print(summary(model, (1, 227, 227)))