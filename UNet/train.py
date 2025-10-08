import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from model import UNet  # Import the UNet model
from utlis import get_loaders  # Import the DataLoader utility function
import numpy as np
import random

# -------------------
# Hyperparameters
# -------------------
LEARNING_RATE = 1e-6
BATCH_SIZE = 8
NUM_EPOCH = 5
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------
# Dataset paths
# -------------------
TRAIN_IMG_DIR = "./dataset/train/small_train"
TRAIN_MASK_DIR = "./dataset/train/small_train_mask"
VAL_IMG_DIR = "./dataset/val/small_train"
VAL_MASK_DIR = "./dataset/val/small_train_mask"

# -------------------
# Image dimensions
# -------------------
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240

# -------------------
# Metrics storage
# -------------------
train_losses = []
val_acc = []
val_dice = []

# -------------------
# Seed setup for reproducibility
# -------------------
seed = random.randint(1, 100)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -------------------
# Training function
# -------------------
def train_fn(loader, model, loss_fn, optimizer, scaler):
    """
    Performs one epoch of training.
    Args:
        loader: DataLoader for training data
        model: PyTorch model
        loss_fn: Loss function
        optimizer: Optimizer
        scaler: GradScaler for mixed precision
    Returns:
        Average training loss for this epoch
    """
    loop = tqdm(loader, leave=False)  # Progress bar
    total_loss = 0.0

    for data, target in loop:
        data = data.to(DEVICE)
        target = target.unsqueeze(1).float().to(DEVICE)

        with torch.amp.autocast(DEVICE):
            predict = model(data)
            loss = loss_fn(predict, target)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(loader)

# -------------------
# Validation/Accuracy function
# -------------------
def check_accuracy(loader, model, DEVICE="cuda"):
    """
    Evaluates model accuracy and Dice coefficient on validation data.
    Args:
        loader: DataLoader for validation data
        model: PyTorch model
        DEVICE: Device to use ("cuda" or "cpu")
    Returns:
        Tuple: (accuracy, mean_dice)
    """
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():  # Disable gradient computation
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.unsqueeze(1).float().to(DEVICE)

            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()         # Convert probabilities to binary mask

            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

            intersection = (preds * y).sum()
            dice_score += (2 * intersection) / (preds.sum() + y.sum() + 1e-8)  # Dice coefficient

    accuracy = num_correct / num_pixels
    mean_dice = dice_score / len(loader)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Mean Dice Score: {mean_dice:.4f}")

    model.train()  # Set model back to training mode
    return float(accuracy), float(mean_dice)

# -------------------
# Main training script
# -------------------
def main():
    # -------------------
    # Training data augmentations
    # -------------------
    train_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        # Resize all images and masks to fixed dimensions
        # Ensures consistent input size for the model

        A.Rotate(limit=35, p=1.0),
        # Randomly rotate the image and mask by up to ±35 degrees
        # p=1.0 means every image will be rotated randomly
        # Helps the model learn rotational invariance

        A.HorizontalFlip(p=0.5),
        # Randomly flip the image and mask horizontally
        # 50% probability, introduces left-right variation
        # Useful for learning symmetry in objects

        A.VerticalFlip(p=0.1),
        # Randomly flip vertically with 10% probability
        # Less frequent vertical flip; useful if orientation varies

        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        # Normalize pixel values to [0,1] range
        # Keeps values consistent for better training stability

        ToTensorV2(),
        # Converts image and mask from numpy arrays to PyTorch tensors
        # Masks remain as single-channel tensors, images as 3-channel
    ])

    # -------------------
    # Validation data preprocessing
    # -------------------
    val_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        # Resize validation images to same size as training

        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        # Normalize validation images

        ToTensorV2(),
        # Convert validation images and masks to PyTorch tensors
    ])

    # Load training and validation DataLoaders
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR, TRAIN_MASK_DIR,
        VAL_IMG_DIR, VAL_MASK_DIR,
        train_transform, val_transform,
        BATCH_SIZE, NUM_WORKERS, PIN_MEMORY
    )

    # Initialize model, loss function, optimizer, and GradScaler
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()                 # Binary cross-entropy with logits
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler(DEVICE)           # GradScaler for mixed precision

    # Training loop over epochs
    for epoch in range(NUM_EPOCH):
        print(f"Epoch [{epoch+1}/{NUM_EPOCH}]")
        train_loss = train_fn(train_loader, model, loss_fn, optimizer, scaler)
        train_losses.append(train_loss)

        acc, dice = check_accuracy(val_loader, model, DEVICE)
        val_acc.append(acc)
        val_dice.append(dice)

# Entry point
if __name__ == "__main__":
    main()
