"""Print vertex and face counts for each unique mesh in the dataset."""
import torch
from pathlib import Path

DATASET_PATH = Path("/projects/gllab/sortur.n/drag_dataset_84k.pt")

dataset = torch.load(DATASET_PATH, weights_only=False)

# Find unique meshes (same logic as the data module)
unique_meshes = []  # (pos, edge_index, first_sample_idx)
mesh_counts = []

for i, data in enumerate(dataset):
    pos = data.pos
    found = False
    for j, (upos, _, _) in enumerate(unique_meshes):
        if pos.shape == upos.shape and torch.allclose(pos, upos, rtol=1e-5, atol=1e-8):
            mesh_counts[j] += 1
            found = True
            break
    if not found:
        unique_meshes.append((pos, data.edge_index, i))
        mesh_counts.append(1)

print(f"{'Mesh':<6} {'Vertices':<10} {'Edges':<10} {'Faces (est)':<12} {'Edges/Node':<12} {'Samples':<8} {'Species'}")
print("-" * 80)

for idx, ((pos, edge_index, sample_idx), count) in enumerate(zip(unique_meshes, mesh_counts)):
    n_verts = pos.size(0)
    n_edges = edge_index.size(1) if edge_index is not None else 0
    edges_per_node = n_edges / max(n_verts, 1)
    # Euler's formula for closed mesh: V - E + F = 2, so F = E - V + 2
    # But edge_index has directed edges (each undirected edge counted twice)
    n_undirected_edges = n_edges // 2
    n_faces_est = n_undirected_edges - n_verts + 2

    species = dataset[sample_idx].species_label if hasattr(dataset[sample_idx], 'species_label') else "N/A"
    print(f"{idx:<6} {n_verts:<10} {n_edges:<10} {n_faces_est:<12} {edges_per_node:<12.1f} {count:<8} {species}")

print(f"\nTotal: {len(unique_meshes)} unique meshes, {sum(mesh_counts)} samples")
