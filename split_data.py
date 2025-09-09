import os, shutil, json
from sklearn.model_selection import train_test_split

def split_dataset(src_dir, dst_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    classes = sorted([d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))])
    os.makedirs(dst_dir, exist_ok=True)

    for split in ['train', 'val', 'test']:
        for cls in classes:
            os.makedirs(os.path.join(dst_dir, split, cls), exist_ok=True)

    for cls in classes:
        src_path = os.path.join(src_dir, cls)
        files = os.listdir(src_path)

        train_files, temp_files = train_test_split(files, test_size=(val_ratio+test_ratio), random_state=seed)
        val_files, test_files = train_test_split(temp_files, test_size=(test_ratio/(val_ratio+test_ratio)), random_state=seed)

        for fname in train_files:
            shutil.copy2(os.path.join(src_path, fname), os.path.join(dst_dir, 'train', cls, fname))
        for fname in val_files:
            shutil.copy2(os.path.join(src_path, fname), os.path.join(dst_dir, 'val', cls, fname))
        for fname in test_files:
            shutil.copy2(os.path.join(src_path, fname), os.path.join(dst_dir, 'test', cls, fname))

    # Save mapping
    class_indices = {cls: i for i, cls in enumerate(classes)}
    with open(os.path.join(dst_dir, "class_indices.json"), "w") as f:
        json.dump(class_indices, f, indent=2)

    print("✅ Split complete! Classes:", class_indices)

if __name__ == "__main__":
    split_dataset("dataset", "dataset_split")
