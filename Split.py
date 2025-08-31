import os
import shutil
import random

# Paths
dataset_dir = "data"   # Your original dataset folder
output_dir = "Dataset"             # New organized dataset folder

# Train/Test split ratio
train_split = 0.8  # 80% train, 20% test

# Classes
classes = ["with_mask", "without_mask"]

# Create train and test folders
for split in ["train", "test"]:
    for cls in classes:
        os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

# Loop through each class folder
for cls in classes:
    class_path = os.path.join(dataset_dir, cls)
    images = os.listdir(class_path)
    random.shuffle(images)

    split_index = int(len(images) * train_split)
    train_images = images[:split_index]
    test_images = images[split_index:]

    # Move train images
    for img in train_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(output_dir, "train", cls, img)
        shutil.copy(src, dst)

    # Move test images
    for img in test_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(output_dir, "test", cls, img)
        shutil.copy(src, dst)

print("✅ Dataset split completed!")