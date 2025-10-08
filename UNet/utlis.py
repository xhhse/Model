from torch.utils.data import DataLoader
from dataset import CarvanaDataset

def get_loaders(
    train_img_dir,
    train_mask_dir,
    val_img_dir,
    val_mask_dir,
    train_transform,
    val_transform,
    batch_size,
    num_workers,
    pin_memory=True
):
    """
    Creates PyTorch DataLoaders for training and validation datasets.
    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
    """

    # -------------------
    # Create training dataset
    # -------------------
    train_ds = CarvanaDataset(
        img_dir=train_img_dir,
        mask_dir=train_mask_dir,
        transform=train_transform
    )

    # -------------------
    # Create validation dataset
    # -------------------
    val_ds = CarvanaDataset(
        img_dir=val_img_dir,
        mask_dir=val_mask_dir,
        transform=val_transform
    )

    # -------------------
    # Create DataLoader for training
    # -------------------
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )

    # -------------------
    # Create DataLoader for validation
    # -------------------
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    # Return both loaders
    return train_loader, val_loader

