import os
import torch
from torch_geometric.datasets import GeometricShapes
import trimesh
import numpy as np

# 1. Setup paths and dataset
output_dir = 'data/stl_files_trimesh' # Using a new folder
os.makedirs(output_dir, exist_ok=True)
dataset = GeometricShapes(root='data/GeometricShapes')

# Class names for descriptive filenames
# class_names = [
#     "Tetrahedron", "Icosahedron", "Cube", "Octahedron", "Dodecahedron",
#     "Cylinder", "Cone", "Sphere", "Torus", "Teapot"
# ]

print(f"Found {len(dataset)} shapes. Converting to STL using trimesh...")

# 2. Iterate through each shape in the dataset
for i, data in enumerate(dataset):
    if data.face is None: # Skip if the graph has no face data
        continue

    # Extract vertices and faces
    vertices = data.pos.numpy()
    faces = data.face.t().numpy()

    # Create a trimesh object directly from vertices and faces
    shape_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # 3. Export the mesh to an STL file
    # print(data.y)
    # shape_name = data.y.item()
    filename = f"{i:02}.stl"
    
    # Use the .export() method to save the file
    # check if watertight, otherwise don't save
    if not shape_mesh.is_watertight:
        print(f"⚠️  Shape {i} is not watertight. Skipping STL export.")
        continue
    shape_mesh.export(os.path.join(output_dir, filename))

print(f"✅ Successfully saved all shapes in the '{output_dir}' directory.")