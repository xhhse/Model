from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def get_loader(batch_size):
    # Define transforms
    train_transforms = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load full training dataset
    full_train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        transform=train_transforms,
        download=True
    )

    # Split train and validation sets
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # Override val_dataset transforms
    val_dataset.dataset.transform = val_transforms

    # Load test dataset
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        transform=val_transforms,
        download=True
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

