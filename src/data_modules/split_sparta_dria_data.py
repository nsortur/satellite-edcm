import os

if __name__ == "__main__":
    # split the data/sparta_dria_h_1123 into train, val, test files and save the resulting files into data/sparta_dria_h_1123_splits
    species_l = ["H", "He", "N", "N2", "O", "O2"]
    for species in species_l:
        files = sorted(os.listdir(f"/home/sortur.n/satellite-edcm/data/sparta_dria/{species}_results_processed"))
        train_files = files[:int(0.8*len(files))]
        val_files = files[int(0.8*len(files)):int(0.9*len(files))]
        test_files = files[int(0.9*len(files)):]
        os.makedirs(f"/home/sortur.n/satellite-edcm/data/sparta_dria_{species}_splits/train", exist_ok=True)
        os.makedirs(f"/home/sortur.n/satellite-edcm/data/sparta_dria_{species}_splits/val", exist_ok=True)
        os.makedirs(f"/home/sortur.n/satellite-edcm/data/sparta_dria_{species}_splits/test", exist_ok=True)
        for f in train_files:
            os.symlink(os.path.abspath(os.path.join(f"/home/sortur.n/satellite-edcm/data/sparta_dria/{species}_results_processed", f)), os.path.join(f"/home/sortur.n/satellite-edcm/data/sparta_dria_{species}_splits/train", f))
        for f in val_files: 
            os.symlink(os.path.abspath(os.path.join(f"/home/sortur.n/satellite-edcm/data/sparta_dria/{species}_results_processed", f)), os.path.join(f"/home/sortur.n/satellite-edcm/data/sparta_dria_{species}_splits/val", f))
            
        for f in test_files:
            os.symlink(os.path.abspath(os.path.join(f"/home/sortur.n/satellite-edcm/data/sparta_dria/{species}_results_processed", f)), os.path.join(f"/home/sortur.n/satellite-edcm/data/sparta_dria_{species}_splits/test", f))
        
