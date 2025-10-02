import torch

from torchvision import transforms
from torch.utils.data import DataLoader, random_split
# from torchvision.datasets import FashionMNIST
from model import GoogLeNet, Inception


def data_process():
    path = "./PetImages"
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 對齊 GoogLeNet 輸入大小
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    full_dataset = full_dataset = datasets.ImageFolder(root=path, transform=transform)
    total_size = len(full_dataset)
    train_size = int(total_size*.8)
    val_size = int(total_size*.1)
    test_size = total_size - train_size - val_size
    print(f"Total: {total_size}, Train: {train_size}, Val: {val_size}, Test: {test_size}")

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # 固定隨機種子，確保可重現
    )
    # batch_size
    bz = 64
    train_loader = DataLoader(train_dataset, batch_size=bz, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=bz, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=bz, shuffle=False)

    return train_loader, val_loader, test_loader

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
    #test_dataloader = test_data_process()
    test_dataloader = torch.load("test_data.pth", weights_only= False)
    num_classes = 2
    model = GoogLeNet(Inception, num_classes)
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
