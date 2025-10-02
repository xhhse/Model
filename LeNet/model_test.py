import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import LeNet
from torch import nn

def test_data_process():
    test_data = FashionMNIST("./Fashion_MNIST",
                         train=False,
                         transform=transforms.Compose([
                             transforms.Resize(size=28),
                             transforms.ToTensor()]),
                         download=True
                         )

    test_dataloader = DataLoader(test_data, batch_size= 128, shuffle= True)

    return test_dataloader

def test_model_process(model, test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    test_corrects = 0
    test_num = 0

    with torch.no_grad():
        model.eval()
        for b_x, b_y in test_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            output = model(b_x)
            pre_label = torch.argmax(output, dim=1)
            test_corrects += torch.sum(pre_label == b_y.data)
            test_num += b_x.size(0)

    print(f"Test Accuracy: {test_corrects.double().item()/test_num: .4f}")

if __name__ == "__main__":
    test_dataloader = test_data_process()
    model = LeNet()
    model.load_state_dict(torch.load("./model/best_model.pth", weights_only = True))
    #test_model_process(model, test_dataloader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    with torch.no_grad():
        model.eval()
        for b_x, b_y in test_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            output = model(b_x)
            pre_lab = torch.argmax(output, dim = 1)
            prob = nn.functional.softmax(model(b_x), dim = 1)
            print(output)
