import os
import glob

def create_splits():
    base_source_dir = "/home/sortur.n/satellite-edcm/data/mesh_data/13_processed"
    base_target_path = "/home/sortur.n/satellite-edcm/data/sparta_dria_{species}_splits"
    
    # List of species to process based on your file naming convention
    species_l = ["He", "H", "N", "N2", "O", "O2"]

    for species in species_l:
        # 1. Find only files that belong to this specific species
        # This matches: drag_force_Output_Processed_SpeciesName_*.txt
        pattern = os.path.join(base_source_dir, f"drag_force_Output_Processed_{species}_*.txt")
        files = sorted(glob.glob(pattern))

        if not files:
            print(f"Warning: No files found for species {species}. Skipping...")
            continue

        # 2. Calculate split indices
        num_files = len(files)
        train_end = int(0.8 * num_files)
        val_end = int(0.9 * num_files)

        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:]

        # 3. Create target directories
        target_dir = base_target_path.format(species=species)
        splits = {"train": train_files, "val": val_files, "test": test_files}

        print(f"Processing {species}: {num_files} files found.")

        for split_name, file_list in splits.items():
            split_path = os.path.join(target_dir, split_name)
            os.makedirs(split_path, exist_ok=True)

            for src_file in file_list:
                # Get the filename (e.g., drag_force_Output_Processed_He_0099.txt)
                file_name = os.path.basename(src_file)
                dest_path = os.path.join(split_path, file_name)

                # 4. Create symlink
                try:
                    # Remove existing link if it exists to avoid FileExistsError
                    if os.path.lexists(dest_path):
                        os.remove(dest_path)
                    
                    os.symlink(os.path.abspath(src_file), dest_path)
                except OSError as e:
                    print(f"Failed to create link for {file_name}: {e}")

if __name__ == "__main__":
    create_splits()
    print("Done! Splits created successfully.")
