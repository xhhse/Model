from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
import numpy as np
from torchvision import transforms

train = FashionMNIST("./Fashion_MNIST",
                     train = True,
                     transform= transforms.Compose([
                         transforms.Resize(size = 28),
                         transforms.ToTensor()]),
                     download= True
                     )
test = FashionMNIST("./Fashion_MNIST",
                     train = False,
                     transform= transforms.Compose([
                         transforms.Resize(size = 28),
                         transforms.ToTensor()]),
                     download= True
                     )

train_loader = DataLoader(dataset=train, batch_size=64, shuffle=True)
for step, (b_x, b_y) in enumerate(train_loader):
    if step > 0:
        break

batch_x = b_x.squeeze().numpy()
batch_y = b_y.numpy()
class_label = train.classes
print(batch_x.shape, b_x.shape)

plt.figure(figsize=(12, 5))
for ii in np.arange(len(batch_y)):
    plt.subplot(4, 16, ii+1)
    plt.imshow(batch_x[ii, :, :], cmap = plt.cm.gray)
    plt.title(class_label[batch_y[ii]], size = 10)
    plt.axis("off")
    plt.subplots_adjust(wspace=0.05)
plt.show()