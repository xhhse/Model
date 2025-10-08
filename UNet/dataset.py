import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image

class CarvanaDataset(Dataset):
    """
    PyTorch Dataset for the Carvana image segmentation dataset.
    Loads images and their corresponding masks, with optional transformations.
    """

    def __init__(self, img_dir, mask_dir, transform=None):
        """
        Args:
            img_dir (str): Path to the directory containing input images.
            mask_dir (str): Path to the directory containing mask images.
            transform (callable, optional): Transformations to apply to both image and mask.
        """
        self.img_dir = img_dir          # Store the image directory path
        self.mask_dir = mask_dir        # Store the mask directory path
        self.transform = transform      # Store the transform function (if provided)
        self.images = os.listdir(self.img_dir)  # List all image filenames in the image directory

    def __len__(self):
        """
        Returns:
            int: Total number of images in the dataset.
        """
        return len(self.images)         # Dataset length is the number of images

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the sample to retrieve.
        Returns:
            tuple: (image, mask), both as numpy arrays or transformed tensors.
        """
        # Construct full path to the image
        img_path = os.path.join(self.img_dir, self.images[index])
        # Construct corresponding mask path by replacing ".jpg" with "_mask.gif"
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))

        # Load the image and convert to RGB (3 channels)
        image = np.array(Image.open(img_path).convert("RGB"))
        # Load the mask and convert to grayscale (1 channel)
        mask = np.array(Image.open(mask_path).convert("L"))

        # Apply transformations if provided
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Return the image and mask pair
        return image, mask
