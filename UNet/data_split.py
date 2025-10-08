import os
import random

# Set directory paths
raw_data_dir = 'dataset/small_train'  # Images directory
mask_data_dir = 'dataset/small_train_mask'  # Masks directory

train_raw_dir = 'dataset/train/small_train'  # Train images directory
val_raw_dir = 'dataset/val/small_train'  # Validation images directory

train_mask_dir = 'dataset/train/small_train_mask'  # Train masks directory
val_mask_dir = 'dataset/val/small_train_mask'  # Validation masks directory

# Set the split ratio (80% train, 20% validation)
train_ratio = 0.8
val_ratio = 0.2

# Ensure that train and validation directories exist
os.makedirs(train_raw_dir, exist_ok=True)
os.makedirs(val_raw_dir, exist_ok=True)
os.makedirs(train_mask_dir, exist_ok=True)
os.makedirs(val_mask_dir, exist_ok=True)

# Get all image filenames from the raw data directory
image_files = os.listdir(raw_data_dir)

# Shuffle the list of image filenames
random.shuffle(image_files)

# Calculate the number of training samples
total_files = len(image_files)
train_size = int(total_files * train_ratio)

# Split the image files into train and validation sets
train_images = image_files[:train_size]
val_images = image_files[train_size:]


# Function to copy images and their corresponding masks
def copy_files(image_list, source_image_dir, source_mask_dir, target_image_dir, target_mask_dir):
    for image in image_list:
        # Copy the image to the appropriate directory
        shutil.copy(os.path.join(source_image_dir, image), os.path.join(target_image_dir, image))

        # Assuming mask image filenames are similar to the image filenames, but with _mask.png as the suffix
        mask_image = image.replace('.jpg', '_mask.gif')  # Modify this if needed to match your mask filenames
        shutil.copy(os.path.join(source_mask_dir, mask_image), os.path.join(target_mask_dir, mask_image))


# Copy training images and masks
copy_files(train_images, raw_data_dir, mask_data_dir, train_raw_dir, train_mask_dir)

# Copy validation images and masks
copy_files(val_images, raw_data_dir, mask_data_dir, val_raw_dir, val_mask_dir)

print(f"Dataset has been split into {len(train_images)} training and {len(val_images)} validation samples.")