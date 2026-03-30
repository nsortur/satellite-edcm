"""
Filter out meshes with more than a given number of nodes from a dataset .pt file.

Usage:
    python scripts/filter_large_meshes.py /path/to/dataset.pt --max-nodes 1000
"""
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


def identify_meshes(dataset):
    """Identify unique meshes and assign each sample a mesh ID."""
    unique_meshes = []  # list of (pos_tensor, mesh_id)
    mesh_assignments = []

    print("Scanning for unique meshes...")
    for data in tqdm(dataset, desc="Identifying meshes"):
        pos = data.pos
        found = False
        for mesh_idx, unique_pos in enumerate(unique_meshes):
            if pos.shape == unique_pos.shape and torch.allclose(pos, unique_pos, rtol=1e-5, atol=1e-8):
                mesh_assignments.append(mesh_idx)
                found = True
                break
        if not found:
            unique_meshes.append(pos)
            mesh_assignments.append(len(unique_meshes) - 1)

    return unique_meshes, mesh_assignments


def main():
    parser = argparse.ArgumentParser(description="Filter large meshes from a PyG dataset .pt file")
    parser.add_argument("input", type=str, help="Path to input .pt dataset file")
    parser.add_argument("--max-nodes", type=int, default=1000, help="Max nodes per mesh to keep (default: 1000)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: same as input)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent

    print(f"Loading dataset from {input_path}...")
    dataset = torch.load(input_path, weights_only=False)
    print(f"Loaded {len(dataset)} samples.")

    # Identify unique meshes
    unique_meshes, mesh_assignments = identify_meshes(dataset)
    num_meshes_before = len(unique_meshes)

    # Compute per-mesh stats
    mesh_node_counts = {i: pos.size(0) for i, pos in enumerate(unique_meshes)}
    mesh_sample_counts = defaultdict(int)
    for mid in mesh_assignments:
        mesh_sample_counts[mid] += 1

    # Print mesh table
    print(f"\nFound {num_meshes_before} unique meshes:")
    print(f"  {'Mesh ID':<10} {'Nodes':<10} {'Samples':<10} {'Status'}")
    print("  " + "-" * 50)

    keep_mesh_ids = set()
    remove_mesh_ids = set()

    for mesh_id in sorted(mesh_node_counts.keys()):
        n_nodes = mesh_node_counts[mesh_id]
        n_samples = mesh_sample_counts[mesh_id]
        status = "KEEP" if n_nodes <= args.max_nodes else "REMOVE"
        print(f"  {mesh_id:<10} {n_nodes:<10} {n_samples:<10} {status}")

        if n_nodes <= args.max_nodes:
            keep_mesh_ids.add(mesh_id)
        else:
            remove_mesh_ids.add(mesh_id)

    num_meshes_after = len(keep_mesh_ids)

    # Filter samples
    filtered_dataset = []
    removed_count = 0
    for data, mesh_id in zip(dataset, mesh_assignments):
        if mesh_id in keep_mesh_ids:
            # Store provenance metadata on each sample
            data.original_mesh_id = mesh_id
            filtered_dataset.append(data)
        else:
            removed_count += 1

    print(f"\n--- Summary ---")
    print(f"  Meshes:  {num_meshes_before} -> {num_meshes_after}")
    print(f"  Samples: {len(dataset)} -> {len(filtered_dataset)} (removed {removed_count})")
    print(f"  Removed mesh IDs: {sorted(remove_mesh_ids)}")
    print(f"  Removed mesh node counts: {[mesh_node_counts[mid] for mid in sorted(remove_mesh_ids)]}")

    # Build output filename
    stem = input_path.stem  # e.g. "drag_dataset_84k"
    output_name = f"{stem}_meshes_{num_meshes_before}_to_{num_meshes_after}.pt"
    output_path = output_dir / output_name

    # Save with metadata dict wrapping the list
    # Store as a dict so we can include filtering metadata
    save_payload = {
        "data_list": filtered_dataset,
        "filter_metadata": {
            "source_file": str(input_path),
            "max_nodes_threshold": args.max_nodes,
            "meshes_before": num_meshes_before,
            "meshes_after": num_meshes_after,
            "removed_mesh_ids": sorted(remove_mesh_ids),
            "removed_mesh_node_counts": {mid: mesh_node_counts[mid] for mid in sorted(remove_mesh_ids)},
            "kept_mesh_ids": sorted(keep_mesh_ids),
            "kept_mesh_node_counts": {mid: mesh_node_counts[mid] for mid in sorted(keep_mesh_ids)},
            "samples_before": len(dataset),
            "samples_after": len(filtered_dataset),
        }
    }

    print(f"\nSaving to {output_path}...")
    torch.save(save_payload, output_path)
    print(f"Done! File size: {output_path.stat().st_size / 1e6:.1f} MB")

    # Also print how to load it
    print(f"\n--- To load ---")
    print(f"  payload = torch.load('{output_path}', weights_only=False)")
    print(f"  dataset = payload['data_list']        # list of PyG Data objects")
    print(f"  metadata = payload['filter_metadata'] # dict with filtering info")


if __name__ == "__main__":
    main()
