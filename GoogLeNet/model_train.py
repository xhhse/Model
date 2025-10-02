import os
import time
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split
# from torchvision.datasets import FashionMNIST
import numpy as np
from torchvision import transforms, datasets
from model import GoogLeNet, Inception
import torch
from torch import nn
import copy
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", message="Truncated File Read")

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

def train_model_process(model, train_dataloader, val_dataloader, epochs, learning_rate = 0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()

    model = model.to(device)

    best_acc = 0.0
    train_loss_all = []
    val_loss_all = []
    train_acc_all = []
    val_acc_all = []

    for epoch in tqdm(range(epochs), desc="Processing"):
        start_time = time.time()
        print(f"Epoch: {epoch+1}/{epochs}")
        print("-"*30)
        train_loss = 0.0
        train_corrects = 0
        val_loss = 0.0
        val_corrects = 0
        train_num = 0
        val_num = 0
        for step, (b_x, b_y) in enumerate(train_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            # model.train()
            output = model(b_x)
            pre_label = torch.argmax(output, dim = 1)
            loss = loss_function(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * b_y.size(0)
            train_corrects += torch.sum(pre_label == b_y.data)
            train_num += b_x.size(0)
        time_use = time.time()-start_time
        print(f"Training time: {time_use//60:.0f}m{time_use%60:.0f}s")
        model.eval()
        for step, (b_x, b_y) in enumerate(val_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            # model.train()
            output = model(b_x)
            pre_label = torch.argmax(output, dim=1)
            loss = loss_function(output, b_y)

            val_loss += loss.item() * b_y.size(0)
            val_corrects += torch.sum(pre_label == b_y.data)
            val_num += b_x.size(0)

        train_loss_all.append(train_loss/train_num)
        val_loss_all.append(val_loss/val_num)
        train_acc_all.append(train_corrects.double().item()/train_num)
        val_acc_all.append(val_corrects.double().item()/val_num)

        print(f"Train Loss: {train_loss_all[-1]:.4f} Train Accuracy: {train_acc_all[-1]:.4f}")
        print(f"Val Loss: {val_loss_all[-1]:.4f} Val Accuracy: {val_acc_all[-1]:.4f}")

        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model = copy.deepcopy(model.state_dict())
    train_process = pd.DataFrame(data={"epoch": range(epochs),
                                       "train_loss_all": train_loss_all,
                                       "val_loss_all": val_loss_all,
                                       "train_acc_all": train_acc_all,
                                       "val_acc_all": val_acc_all})
    folder_name = "model"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    torch.save(best_model, './model/best_model.pth')
    return train_process

def matplot_acc_loss(dataframe):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(dataframe["epoch"], dataframe["train_loss_all"], "ro-", label = "train loss")
    plt.plot(dataframe["epoch"], dataframe["val_loss_all"], "bs-", label="val loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    plt.subplot(1, 2, 2)
    plt.plot(dataframe["epoch"], dataframe["train_acc_all"], "ro-", label = "train acc")
    plt.plot(dataframe["epoch"], dataframe["val_acc_all"], "bs-", label="val acc")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    plt.show()

if __name__ == "__main__":
    num_classes = 2
    model = GoogLeNet(Inception, 2)
    train_dataloader, val_dataloader, test_dataloader= data_process()
    torch.save(test_dataloader, "test_data.pth")
    train_df = train_model_process(model, train_dataloader, val_dataloader, 10, learning_rate=0.001)
    matplot_acc_loss(train_df)
