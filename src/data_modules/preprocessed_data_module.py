import torch
import random
from torch.utils.data import Dataset, Sampler
from torch_geometric.loader import DataLoader as GeometricDataLoader
import pytorch_lightning as pl
from tqdm import tqdm


class MaxNodesBatchSampler(Sampler):
    """
    Groups graphs into batches so the total number of nodes per batch
    never exceeds `max_nodes`. This keeps memory usage consistent regardless
    of graph size — small graphs get many per batch, large graphs go solo.
    """
    def __init__(self, dataset, max_nodes, shuffle=True):
        self.dataset = dataset
        self.max_nodes = max_nodes
        self.shuffle = shuffle

        # Pre-compute node counts from the raw data_list (before __getitem__ transform)
        self.node_counts = []
        for i in range(len(dataset)):
            data = dataset.data_list[i]
            self.node_counts.append(data.pos.size(0))

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        batch = []
        batch_nodes = 0
        for idx in indices:
            node_count = self.node_counts[idx]

            # If adding this graph would exceed the budget, yield current batch first
            if batch and (batch_nodes + node_count > self.max_nodes):
                yield batch
                batch = []
                batch_nodes = 0

            batch.append(idx)
            batch_nodes += node_count

        # Don't forget the last batch
        if batch:
            yield batch

    def __len__(self):
        # Estimate batch count (recomputed each epoch due to shuffle)
        count = 0
        batch_nodes = 0
        for nc in self.node_counts:
            if batch_nodes + nc > self.max_nodes and batch_nodes > 0:
                count += 1
                batch_nodes = 0
            batch_nodes += nc
        if batch_nodes > 0:
            count += 1
        return count


class PreprocessedDragDataset(Dataset):
    """
    Dataset wrapper that extracts the 8-feature split and applies normalization.
    """
    def __init__(self, data_list, feature_min=None, feature_max=None, norm_features=True):
        super().__init__()
        self.data_list = data_list
        self.feature_min = feature_min
        self.feature_max = feature_max
        self.norm_features = norm_features

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx].clone()
        gf = data.global_features.squeeze()

        # Extract 8-feature split
        node_features = gf[0:8]
        orientation = gf[8:10]

        # Normalize based on training stats
        if self.norm_features and self.feature_min is not None and self.feature_max is not None:
            denom = (self.feature_max - self.feature_min).clamp(min=1e-8)
            node_features = (node_features - self.feature_min) / denom

        # Assign to PyG variables
        num_nodes = data.pos.size(0)
        data.x = node_features.repeat(num_nodes, 1).to(torch.float32)
        data.orientation = orientation.unsqueeze(0).to(torch.float32)

        if hasattr(data, 'y') and data.y is not None:
            data.y = data.y.view(1).to(torch.float32)

        # Cleanup raw fields to save memory
        if hasattr(data, 'global_features'):
            del data.global_features
        if hasattr(data, 'species_label'):
            del data.species_label

        return data


class PreprocessedDragDataModule(pl.LightningDataModule):
    """
    LightningDataModule handling deterministic mesh-based splitting.
    Uses dynamic batching based on total node count to prevent OOM errors.
    """
    def __init__(
        self,
        data_dir: str = "/projects/gllab/sortur.n/drag_dataset_84k.pt",
        batch_size: int = 64,
        max_nodes_per_batch: int = 4000,
        num_workers: int = 4,
        validation_meshes: list = None,
        norm_features: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_nodes_per_batch = max_nodes_per_batch
        self.num_workers = num_workers
        self.validation_meshes = validation_meshes if validation_meshes is not None else [0, 1, 2, 3, 4]
        self.norm_features = norm_features

        self.data_train = None
        self.data_val = None

    def setup(self, stage: str = None):
        if self.data_train is not None:
            return

        print(f"Loading preprocessed dataset from {self.data_dir}...")
        full_list = torch.load(self.data_dir, weights_only=False)
        
        # 1. Identify Unique Meshes
        unique_meshes = []
        mesh_assignments = []

        print("Identifying unique meshes to split datasets safely...")
        for data in tqdm(full_list, desc="Scanning geometries"):
            pos = data.pos
            found_match = False
            for mesh_idx, unique_pos in enumerate(unique_meshes):
                if pos.shape == unique_pos.shape:
                    if torch.allclose(pos, unique_pos, rtol=1e-5, atol=1e-8):
                        mesh_assignments.append(mesh_idx)
                        found_match = True
                        break
            
            if not found_match:
                unique_meshes.append(pos)
                mesh_assignments.append(len(unique_meshes) - 1)

        # Attach mesh numbers to data
        for i, data in enumerate(full_list):
            data.mesh_number = mesh_assignments[i]

        print(f"\nFound {len(unique_meshes)} unique base meshes.")
        print(f"Assigning meshes {self.validation_meshes} to validation.")

        # 2. Split by user-specified MESH ID
        train_list = [d for d in full_list if d.mesh_number not in self.validation_meshes]
        val_list = [d for d in full_list if d.mesh_number in self.validation_meshes]

        print(f"Dataset split results:")
        print(f"  Train: {len(train_list)} samples")
        print(f"  Val:   {len(val_list)} samples")

        # 3. Calculate Normalization Bounds (Strictly from Training Split)
        feature_min, feature_max = None, None
        if self.norm_features:
            print("Calculating 8-feature normalization bounds from training set only...")
            train_gfs = torch.stack([d.global_features.squeeze()[0:8] for d in train_list])
            feature_min = train_gfs.min(dim=0)[0]
            feature_max = train_gfs.max(dim=0)[0]

        # 4. Instantiate Datasets
        self.data_train = PreprocessedDragDataset(train_list, feature_min, feature_max, self.norm_features)
        self.data_val = PreprocessedDragDataset(val_list, feature_min, feature_max, self.norm_features)

        # 5. Log dynamic batching stats
        train_sampler = MaxNodesBatchSampler(self.data_train, self.max_nodes_per_batch, shuffle=False)
        print(f"\n  Dynamic batching (max_nodes_per_batch={self.max_nodes_per_batch}):")
        print(f"    Estimated train batches/epoch: {len(train_sampler)}")
        val_sampler = MaxNodesBatchSampler(self.data_val, self.max_nodes_per_batch, shuffle=False)
        print(f"    Estimated val batches/epoch:   {len(val_sampler)}")

    def train_dataloader(self):
        sampler = MaxNodesBatchSampler(
            self.data_train, self.max_nodes_per_batch, shuffle=True
        )
        return GeometricDataLoader(
            self.data_train, batch_sampler=sampler,
            num_workers=self.num_workers, persistent_workers=(self.num_workers > 0), pin_memory=True
        )

    def val_dataloader(self):
        sampler = MaxNodesBatchSampler(
            self.data_val, self.max_nodes_per_batch, shuffle=False
        )
        return GeometricDataLoader(
            self.data_val, batch_sampler=sampler,
            num_workers=self.num_workers, persistent_workers=(self.num_workers > 0), pin_memory=True
        )
