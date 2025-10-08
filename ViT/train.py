import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import get_loader
from model import ViT

# -------------------------------
# Training function
# -------------------------------
def train_model(model, train_loader, val_loader, device, epochs=10, lr=1e-4, weight_decay=0,
                betas=(0.9, 0.999), checkpoint_path="best_model.pth"):
    """
    Train the model, save the best model based on validation accuracy, and record history.
    Returns:
        history: dict of train_loss, val_loss, train_acc, val_acc
        best_val_acc: best validation accuracy
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)

    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in tqdm(range(epochs), position=0, leave=True):
        # -------------------------------
        # Training
        # -------------------------------
        model.train()
        total_loss, total_correct = 0, 0

        for images, labels in tqdm(train_loader, position=0, leave=True):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += (outputs.argmax(1) == labels).sum().item()

        train_loss = total_loss / len(train_loader)
        train_acc = total_correct / len(train_loader.dataset)

        # -------------------------------
        # Validation
        # -------------------------------
        model.eval()
        val_loss_total, val_correct = 0, 0
        val_total_samples = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, position=0, leave=True):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss_total += loss.item()
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total_samples += labels.size(0)

        val_loss = val_loss_total / len(val_loader)
        val_acc = val_correct / val_total_samples

        # -------------------------------
        # Save best model
        # -------------------------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)

        # -------------------------------
        # Record history
        # -------------------------------
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

    print(f"Training complete. Best validation accuracy: {best_val_acc*100:.2f}%")
    return history, best_val_acc

# -------------------------------
# Test function
# -------------------------------
def test_model(model, test_loader, device, checkpoint_path=None):
    """
    Evaluate the model on test dataset. Optionally load a checkpoint.
    """
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded model weights from {checkpoint_path}")

    model.to(device)
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, position=0, leave=True):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    test_acc = correct / total
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    return test_acc

# -------------------------------
# Plot training history
# -------------------------------
def plot_training_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12,5))

    # Loss
    plt.subplot(1,2,1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()

    # Accuracy
    plt.subplot(1,2,2)
    plt.plot(epochs, [x*100 for x in history["train_acc"]], label="Train Acc")
    plt.plot(epochs, [x*100 for x in history["val_acc"]], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training & Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


# -------------------------------
# Main execution
# -------------------------------
if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 64
    EPOCHS = 10

    # Load MNIST dataset
    train_loader, val_loader, test_loader = get_loader(batch_size=BATCH_SIZE)

    # Initialize ViT model
    IN_CHANNELS = 1
    IMAGE_SIZE = 28
    PATCH_SIZE = 4
    EMBED_DIM = PATCH_SIZE ** 2 * IN_CHANNELS
    NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
    DROPOUT = 0.001
    NUM_HEADS = 8
    ACTIVATION = "gelu"
    NUM_ENCODERS = 12
    NUM_CLASSES = 10

    model = ViT(IN_CHANNELS, PATCH_SIZE, EMBED_DIM, NUM_PATCHES,
                DROPOUT, NUM_HEADS, ACTIVATION, NUM_ENCODERS, NUM_CLASSES).to(DEVICE)

    # Train model
    history, best_val_acc = train_model(model, train_loader, val_loader, DEVICE,
                                        epochs=EPOCHS, lr=1e-4, checkpoint_path="best_vit_mnist.pth")

    # Plot training curves
    plot_training_history(history)

    # Test model
    test_acc = test_model(model, test_loader, DEVICE, checkpoint_path="best_vit_mnist.pth")
