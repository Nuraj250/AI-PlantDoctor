import os
import shutil
from sklearn.model_selection import train_test_split
from glob import glob

def split_dataset(raw_dir="data/raw/PlantVillage", output_dir="data", split_ratio=0.8):
    os.makedirs(f"{output_dir}/train", exist_ok=True)
    os.makedirs(f"{output_dir}/val", exist_ok=True)

    class_dirs = [d for d in glob(f"{raw_dir}/*") if os.path.isdir(d)]

    for class_path in class_dirs:
        class_name = os.path.basename(class_path)
        image_paths = glob(os.path.join(class_path, "*.*"))

        # Split images
        train_imgs, val_imgs = train_test_split(image_paths, train_size=split_ratio, random_state=42)

        # Copy to train
        for img in train_imgs:
            dest_dir = os.path.join(output_dir, "train", class_name)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(img, dest_dir)

        # Copy to val
        for img in val_imgs:
            dest_dir = os.path.join(output_dir, "val", class_name)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(img, dest_dir)

    print(f"✅ Dataset split complete: {len(class_dirs)} classes processed.")
    print(f"➡️  Output in: {output_dir}/train and {output_dir}/val")

if __name__ == "__main__":
    split_dataset()
