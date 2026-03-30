import torch
import os
import glob
import numpy as np
from omegaconf import OmegaConf
from hydra.utils import instantiate
from math import sqrt
from tqdm import tqdm
import sys
# --- Configuration ---
BASE_RUN_PATH = "multirun/2026-03-16/19-31-11/"
version_no = "5122023"
DATA_BASE_DIR = "data"

SPECIES_MAP = {
    0: "H",
    1: "He",
    2: "N",
    3: "N2",
    4: "O",
    5: "O2"
}

def evaluate_species(species_id, species_name):
    print(f"\n{'='*30}")
    print(f"  Evaluating Species: {species_name} (ID: {species_id})")
    print(f"{'='*30}")
    
    # ---------------------------------------------------------
    # 1. Setup Paths
    # ---------------------------------------------------------
    run_dir = os.path.join(BASE_RUN_PATH, str(species_id))
    config_path = os.path.join(run_dir, ".hydra", "config.yaml")
    # Note: Using version_0/checkpoints/last.ckpt based on your description
    ckpt_path = os.path.join(run_dir, "lightning_logs", f"version_{version_no}", "checkpoints", "last.ckpt")
    species_data_dir = os.path.join(DATA_BASE_DIR, f"sparta_dria_{species_name}_splits")

    # Verify existence
    if not os.path.exists(config_path):
        print(f"[Skip] Config not found: {config_path}")
        return None
    if not os.path.exists(ckpt_path):
        print(f"[Skip] Checkpoint not found: {ckpt_path}")
        return None

    # ---------------------------------------------------------
    # 2. Data Preparation (Normalization Stats)
    # ---------------------------------------------------------
    # We must instantiate the DataModule to calculate the min/max 
    # from the TRAIN set so we can un-normalize the TEST set correctly.
    print(f"loading data stats from: {species_data_dir}")
    
    # Dynamic import to ensure it works from root
    try:
        from src.data_modules.sparta_dria_tensor_dataset import SpartaDRIADragDataModule
    except ImportError:
        print("Error: Could not import `src.data_modules...`. Run this from project root.")
        return None

    # Load Config to get batch size
    cfg = OmegaConf.load(config_path)
    
    dm = SpartaDRIADragDataModule(
        data_dir=species_data_dir,
        batch_size=128,
        num_workers=1
    )
    
    # setup('test') triggers the calculation of norm_stats on training data
    dm.setup(stage='test')
    test_loader = dm.test_dataloader()

    # Retrieve De-Normalization Stats for Drag Coefficient
    # We need these to convert MSE back to real units
    try:
        drag_stats = dm.test_dataset.norm_stats['drag_coeff']
        y_min = drag_stats['min']
        y_max = drag_stats['max']
        print(f"  > Normalization Stats [Drag]: Min={y_min:.4f}, Max={y_max:.4f}")
    except KeyError:
        print("  > Error: 'drag_coeff' not found in norm_stats.")
        return None

    # ---------------------------------------------------------
    # 3. Model Instantiation & Weight Loading
    # ---------------------------------------------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("  > Instantiating Model...")
    try:
        # Instantiate the underlying network (e.g., EquiMLP)
        model = instantiate(cfg.model)
        model.to(device)
    except Exception as e:
        print(f"  > Error instantiating model: {e}")
        return None

    print(f"  > Loading Checkpoint: {ckpt_path}")
    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = checkpoint['state_dict']
        
        # --- FIX BASED ON CDTrainingModule ---
        # The LightningModule wraps the model in 'self.net'. 
        # Therefore, keys in state_dict start with 'net.'. 
        # We must strip this prefix to load into the raw model.
        new_state_dict = state_dict
        # for k, v in state_dict.items():
        #     if k.startswith('net.'):
        #         new_state_dict[k[4:]] = v  # Strip 'net.'
        #     else:
        #         new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict)
        model.eval()
    except Exception as e:
        print(f"  > Error loading weights: {e}")
        return None

    # ---------------------------------------------------------
    # 4. Evaluation Loop
    # ---------------------------------------------------------
    all_errors = [] # List to store (prediction - truth) in REAL units

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"  > Inference {species_name}", leave=False):
            # The dataset returns: features, geometric_features, target
            if len(batch) == 3:
                features, geo_features, targets = batch
                
                features = features.to(device)
                geo_features = geo_features.to(device)
                targets = targets.to(device)

                # --- FIX BASED ON CDTrainingModule ---
                # _shared_step calls: preds = self((x, geo_x))
                # So we pass a TUPLE to the model
                try:
                    outputs = model((features, geo_features)).squeeze()
                except Exception as e:
                    # Fallback: some models might take concatenated inputs or separate args
                    # But based on your code, tuple is the intended path.
                    print(f"  > Inference Error: {e}")
                    return None
            else:
                print(f"  > Unexpected batch length: {len(batch)}")
                continue

            # --- Un-normalize ---
            # Prediction and Target are both in [0, 1] range.
            # Convert both to Real Units before comparing.
            # Formula: val_real = val_norm * (max - min) + min
            
            preds_real = outputs * (y_max - y_min) + y_min
            targets_real = targets * (y_max - y_min) + y_min
            
            # Calculate residual (Error)
            error = preds_real - targets_real
            all_errors.append(error.cpu())

    # ---------------------------------------------------------
    # 5. Calculate Metrics
    # ---------------------------------------------------------
    if not all_errors:
        return None

    # Concatenate all batches
    all_errors_tensor = torch.cat(all_errors)
    
    # 1. RMSE (Root Mean Squared Error)
    mse = torch.mean(all_errors_tensor ** 2).item()
    rmse = sqrt(mse)
    
    # 2. Std (Standard Deviation of the Errors)
    # This tells you the spread of the error distribution
    std_error = torch.std(all_errors_tensor).item()
    
    print(f"  > Result: RMSE={rmse:.5f}, Std={std_error:.5f}")
    
    return {
        "species": species_name,
        "rmse": rmse,
        "std_error": std_error
    }

def main():
    results = []
    
    # Iterate over all species
    for species_id in range(6):
        if species_id in SPECIES_MAP:
            name = SPECIES_MAP[species_id]
            res = evaluate_species(species_id, name)
            if res:
                results.append(res)
    
    # Final Table
    print("\n\n")
    print("="*55)
    print(f"{'FINAL METRICS SUMMARY (Real Units)':^55}")
    print("="*55)
    print(f" {'Species':<10} | {'RMSE':<18} | {'Std Dev (Error)':<18}")
    print("-" * 55)
    
    avg_rmse = 0
    for res in results:
        print(f" {res['species']:<10} | {res['rmse']:<18.5f} | {res['std_error']:<18.5f}")
        avg_rmse += res['rmse']
        
    if results:
        print("-" * 55)
        print(f" {'AVERAGE':<10} | {avg_rmse/len(results):<18.5f} | {'-':<18}")
    print("="*55)

if __name__ == "__main__":
    main()
