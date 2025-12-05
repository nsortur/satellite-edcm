import os

if __name__ == "__main__":
    # split the data/sparta_dria_h_1123 into train, val, test files and save the resulting files into data/sparta_dria_h_1123_splits
    files = sorted(os.listdir("data/sparta_dria_h_1123"))
    train_files = files[:int(0.7*len(files))]
    val_files = files[int(0.7*len(files)):int(0.85*len(files))]
    test_files = files[int(0.85*len(files)):]
    os.makedirs("data/sparta_dria_h_1123_splits/train", exist_ok=True)
    os.makedirs("data/sparta_dria_h_1123_splits/val", exist_ok=True)
    os.makedirs("data/sparta_dria_h_1123_splits/test", exist_ok=True)
    for f in train_files:
        os.symlink(os.path.abspath(os.path.join("data/sparta_dria_h_1123", f)), os.path.join("data/sparta_dria_h_1123_splits/train", f))
    for f in val_files: 
        os.symlink(os.path.abspath(os.path.join("data/sparta_dria_h_1123", f)), os.path.join("data/sparta_dria_h_1123_splits/val", f))
        
    for f in test_files:
        os.symlink(os.path.abspath(os.path.join("data/sparta_dria_h_1123", f)), os.path.join("data/sparta_dria_h_1123_splits/test", f))
    