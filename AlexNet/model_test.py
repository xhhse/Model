import torch

from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from model import AlexNet
from torch import nn

def test_data_process():
    test_data = FashionMNIST("./Fashion_MNIST",
                         train=False,
                         transform=transforms.Compose([
                             transforms.Resize(size=227),
                             transforms.ToTensor()]),
                         download=True
                         )

    test_dataloader = DataLoader(test_data, batch_size= 128, shuffle= True)

    return test_dataloader

def confusion_matrix(preds, labels, num_classes):

    preds = preds.view(-1)
    labels = labels.view(-1)

    # 初始化混淆矩陣
    conf_mat = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    for p, t in zip(preds, labels):
        conf_mat[t.long(), p.long()] += 1

    return conf_mat

def test_model_process(model, test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    test_corrects = 0
    test_num = 0
    true_label = torch.tensor([])
    pred_label = torch.tensor([])
    with torch.no_grad():
        model.eval()
        for b_x, b_y in test_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            output = model(b_x)
            pre_label = torch.argmax(output, dim=1)
            test_corrects += torch.sum(pre_label == b_y.data)
            test_num += b_x.size(0)
            true_label = torch.concat((true_label, b_y.data.cpu()), dim = 0)
            pred_label = torch.concat((pred_label, pre_label.cpu()))
    num_classes = len(set(torch.unique(true_label).tolist()))
    cm = confusion_matrix(pred_label, true_label, num_classes)
    print(cm)
    print(f"Test Accuracy: {test_corrects.double().item()/test_num: .4f}")

if __name__ == "__main__":
    test_dataloader = test_data_process()
    model = AlexNet()
    model.load_state_dict(torch.load("./model/best_model.pth", weights_only = True))
    test_model_process(model, test_dataloader)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # with torch.no_grad():
    #     model.eval()
    #     for b_x, b_y in test_dataloader:
    #         b_x = b_x.to(device)
    #         b_y = b_y.to(device)
    #         output = model(b_x)
    #         pre_lab = torch.argmax(output, dim = 1)
    #         prob = nn.functional.softmax(model(b_x), dim = 1)
    #         print(output)
