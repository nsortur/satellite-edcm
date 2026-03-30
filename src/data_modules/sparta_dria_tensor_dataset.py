from typing import Any
from pathlib import Path
import hydra

import pytorch_lightning as pl
from torch.utils.data import Dataset
from torchvision.transforms import transforms as T
import torch
import re
import numpy as np
import os

class SpartaDRIADragDataset(Dataset):
    def __init__(self, data_dir: Path, stage: str = None, norm_stats: dict = None):
        self.data_dir = hydra.utils.to_absolute_path(data_dir)
        #self.data_dir = data_dir
        self.stage = stage
        self.file_paths = []
        self.metadata = []
        
        # Normalization statistics
        self.norm_stats = norm_stats
        
        self._load_file_paths()
        self._cache_metadata()
    
    def _load_file_paths(self):
        # If stage is provided, load from the appropriate subdirectory
        if self.stage in ['train', 'val', 'test']:
            dir_path = os.path.join(self.data_dir, self.stage)
        else:
            dir_path = self.data_dir
            
        for filename in sorted(os.listdir(dir_path)):
            if filename.endswith('.txt') and 'Output_Processed' in filename:
                self.file_paths.append(os.path.join(dir_path, filename))
    
    def _parse_file(self, file_path):
        with open(file_path, 'r') as f:
            content = f.read()
        
        patterns = {
            'drag_coeff': r'Resulting Coefficient of Drag: ([\d\.\-e\+]+)',
            'velocity': r'Free-Stream Velocity: \[([\d\.\-e\+, ]+)\] m/s',
            'orientation': r'Orientation: \[([\d\.\-e\+, ]+)\]',
            'accomodation': r'Coefficient of Accomodation: ([\d\.\-e\+]+)',
            'temperature': r'Temperature: ([\d\.\-e\+]+) K'
        }
        
        data = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                if key == 'velocity':
                    # Parse velocity into x, y, z components
                    values = [float(x.strip()) for x in match.group(1).split(',')]
                    data['vel_x'] = values[0]
                    data['vel_y'] = values[1]
                    data['vel_z'] = values[2]
                elif key == 'orientation':
                    # Parse orientation into x, y, z components
                    values = [float(x.strip()) for x in match.group(1).split(',')]
                    data['orient_x'] = values[0]
                    data['orient_y'] = values[1]
                    data['orient_z'] = values[2]
                else:
                    # Parse single values
                    data[key] = float(match.group(1))
            else:
                print(f"No match found for {key}")
                print(f"Pattern: {pattern}")
                print(f"Content snippet: {content[:500]}")
        
        return data
    
    def _normalize_data(self, data, exclude_keys=None):
        data_norm = {}
        for key, value in data.items():
            if exclude_keys and key in exclude_keys:
                data_norm[key] = value
            else:
                data_norm[key] = (value - self.norm_stats[key]['min']) / (self.norm_stats[key]['max'] - 
                                                                      self.norm_stats[key]['min'])
        return data_norm
    
    def _cache_metadata(self):
        print(f"Caching metadata for {len(self.file_paths)} files...")
        for file_path in self.file_paths:
            metadata = self._parse_file(file_path)
            self.metadata.append(metadata)
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        if hasattr(self, 'metadata') and self.metadata:
            data = self.metadata[idx]
        else:
            data = self._parse_file(self.file_paths[idx])
        
        # exclude orientation - already unit vector
        data = self._normalize_data(data, exclude_keys=['orient_x', 'orient_y', 'orient_z']) if self.norm_stats else data
        
        # Convert to tensors
        features = torch.tensor([
            data['vel_x'],               # Velocity components
            data['vel_y'],
            data['vel_z'],
            data['accomodation'],        # 1 value
            data['temperature']          # 1 value
        ], dtype=torch.float32)
        
        geometric_features = torch.tensor([
            data['orient_x'],            # Orientation components
            data['orient_y'],
            data['orient_z']
        ], dtype=torch.float32)
        target = torch.tensor(data['drag_coeff'], dtype=torch.float32)
                
        return features, geometric_features, target
    
    
class SpartaDRIADragDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: Path, batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def _compute_normalization_stats(self):
        print("Computing normalization statistics from training data...")
        
        norm_stats = {}
        
        features_list = []
        geometric_list = []
        targets_list = []
        
        # Create temporary train dataset without normalization
        temp_train = SpartaDRIADragDataset(data_dir=self.data_dir, stage='train')
        for i in range(len(temp_train)):
            features, geometric_features, target = temp_train[i]
            features_list.append(features.numpy())
            geometric_list.append(geometric_features.numpy())
            targets_list.append(target.numpy())
            
        features_array = np.array(features_list)
        geometric_array = np.array(geometric_list)
        targets_array = np.array(targets_list)
        all_data = np.hstack((features_array, geometric_array, targets_array.reshape(-1, 1)))
        feature_names = ['vel_x', 'vel_y', 'vel_z', 'accomodation', 'temperature',
                         'orient_x', 'orient_y', 'orient_z', 'drag_coeff']
        for i, name in enumerate(feature_names):
            norm_stats[name] = {
                'min': np.min(all_data[:, i]),
                'max': np.max(all_data[:, i])
            }
        
        return norm_stats
    
    def setup(self, stage: str = None):
        norm_stats = self._compute_normalization_stats()
        
        if stage == 'fit' or stage is None:
            self.train_dataset = SpartaDRIADragDataset(data_dir=self.data_dir, 
                                                       stage='train', norm_stats=norm_stats)
            self.val_dataset = SpartaDRIADragDataset(data_dir=self.data_dir, 
                                                     stage='val', norm_stats=norm_stats)
        
        if stage == 'test' or stage is None:
            self.test_dataset = SpartaDRIADragDataset(data_dir=self.data_dir, 
                                                      stage='test', norm_stats=norm_stats)
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                                           persistent_workers=True)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                                           persistent_workers=True)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                                           persistent_workers=True)


# run a quick test
if __name__ == "__main__":
    
    # test data module
    data_module = SpartaDRIADragDataModule(data_dir="data/sparta_dria_splits", batch_size=1)
    data_module.setup(stage='test')
    loader_test = data_module.test_dataloader()
    for batch in loader_test:
        features, geometric_features, targets = batch
        # print(f"Batch features shape: {features.shape}")
        # print(f"Batch geometric features shape: {geometric_features.shape}")
        # print(f"Batch targets shape: {targets.shape}")
        print("Features:", features)
        print("Geometric Features:", geometric_features)
        print("Targets:", targets)
        break
    
    
