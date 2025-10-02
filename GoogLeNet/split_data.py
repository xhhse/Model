import os
import shutil
import random

def split_dataset(root_dir, output_dir, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    將 root_dir 下的子資料夾 (類別) 依比例分割成 train/val/test，並輸出到 output_dir
    """
    random.seed(seed)

    # 建立輸出資料夾
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    # 遍歷每個類別資料夾
    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue  # 跳過非資料夾

        images = [f for f in os.listdir(class_path)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        random.shuffle(images)

        total = len(images)
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)

        splits = {
            "train": images[:train_count],
            "val": images[train_count:train_count + val_count],
            "test": images[train_count + val_count:]
        }

        # 把檔案複製到對應資料夾
        for split, split_files in splits.items():
            split_class_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(split_class_dir, exist_ok=True)
            for file in split_files:
                src = os.path.join(class_path, file)
                dst = os.path.join(split_class_dir, file)
                shutil.copy2(src, dst)

    print(f"資料分割完成！train/val/test 資料夾已輸出到 {output_dir}")

# 使用範例
if __name__ == "__main__":
    input_folder = "PetImages"   # 原始資料集
    output_folder = "Cat_Dog"  # 分割後輸出資料集
    split_dataset(input_folder, output_folder, train_ratio=0.9, val_ratio=0.1)