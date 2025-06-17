import os
import re
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def extract_cd(file_content):
    for line in file_content.splitlines():
        if line.startswith("CD"):
            match = re.search(r"CD\s*=\s*([\d\.eE+-]+)", line)
            if match:
                return match.group(1)
    return "N/A"

def print_orientations_with_cd(species="N2", blocks_to_use=None):
    orientations_file = os.path.join("rsm_sparta_data", "orientations.txt")
    results_dir = os.path.join("rsm_sparta_data", f"{species}_Results")
    
    with open(orientations_file, "r") as f:
        content = f.read()
    
    blocks = [blk.strip() for blk in re.split(r'\n\s*\n', content) if blk.strip()]
    
    for b, block in enumerate(blocks, start=1):
        if blocks_to_use and b not in blocks_to_use:
            continue
        orientation_lines = [line.strip() for line in block.splitlines() if line.strip()]
        for i, orientation in enumerate(orientation_lines, start=1):
            file_name = f"drag_force_N2_{b}_{i}.txt"
            file_path = os.path.join(results_dir, file_name)
            try:
                with open(file_path, "r") as df:
                    file_data = df.read()
                    cd_value = extract_cd(file_data)
            except FileNotFoundError:
                cd_value = "File not found"
            print(f"{orientation} -> CD: {cd_value}")

def plot_orientations_3d(species="N2", blocks_to_use=None, plotmode="sparta"):
    if plotmode == "sparta":
        orientations_file = os.path.join("rsm_sparta_data", "orientations.txt")
        results_dir = os.path.join("rsm_sparta_data", f"{species}_Results")
        
        with open(orientations_file, "r") as f:
            content = f.read()
            
        blocks = [blk.strip() for blk in re.split(r'\n\s*\n', content) if blk.strip()]
        
        xs, ys, zs, cds = [], [], [], []
        for b, block in enumerate(blocks, start=1):
            if blocks_to_use and b not in blocks_to_use:
                continue
            orientation_lines = [line.strip() for line in block.splitlines() if line.strip()]
            for i, orientation in enumerate(orientation_lines, start=1):
                # Remove square brackets and split the coordinates.
                inner = orientation[1:-1].strip()
                if ',' in inner:
                    parts = [p.strip() for p in inner.split(',') if p.strip()]
                else:
                    parts = inner.split()
                if len(parts) == 3:
                    try:
                        x, y, z = map(float, parts)
                    except ValueError:
                        continue
                else:
                    continue

                file_name = f"drag_force_N2_{b}_{i}.txt"
                file_path = os.path.join(results_dir, file_name)
                try:
                    with open(file_path, "r") as df:
                        file_data = df.read()
                        cd_value = extract_cd(file_data)
                except FileNotFoundError:
                    continue
                
                try:
                    cd_float = float(cd_value)
                except ValueError:
                    cd_float = 0.0

                xs.append(x)
                ys.append(y)
                zs.append(z)
                cds.append(cd_float)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(xs, ys, zs, c=cds, cmap='viridis',  marker='o', s=60)
        print(np.std(cds))
        ax.set_title(f'{species} Orientation Vectors Colored by CD Value (sparta)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        plt.colorbar(sc, ax=ax, label='CD Value')
        plt.show()
    
    elif plotmode == "rsm":
        # RSM mode: each orientation vector from orientations.txt is paired with the corresponding line in rsm_n2_results.dat.
        # Both files are split into blocks using blank lines, then filtered by --blocks if provided.
        orientations_file = os.path.join("rsm_sparta_data", "orientations.txt")
        rsm_file = os.path.join("rsm_sparta_data", "rsm_n2_results.dat")
        
        with open(orientations_file, "r") as f:
            orientation_content = f.read()
        orientation_blocks = [blk.strip() for blk in re.split(r'\n\s*\n', orientation_content) if blk.strip()]
        
        with open(rsm_file, "r") as f:
            rsm_content = f.read()
        rsm_blocks = [blk.strip() for blk in re.split(r'\n\s*\n', rsm_content) 
                      if blk.strip() and not blk.lstrip().startswith("//")]
        
        
        if len(orientation_blocks) != len(rsm_blocks):
            print("Mismatch between number of orientation blocks and rsm blocks.")
            return
        
        xs, ys, zs, cds = [], [], [], []
        for b, (op_block, rsm_block) in enumerate(zip(orientation_blocks, rsm_blocks), start=1):
            if blocks_to_use and b not in blocks_to_use:
                continue
            orientation_lines = [line.strip() for line in op_block.splitlines() if line.strip()]
            rsm_lines = [line.strip() for line in rsm_block.splitlines() if line.strip()]
            if len(orientation_lines) != len(rsm_lines):
                print(len(orientation_lines))
                print(len(rsm_lines))
                print(f"Block {b}: Mismatch in number of lines between orientations and rsm data.")
                continue
            for orient_line, rsm_line in zip(orientation_lines, rsm_lines):
                # Parse orientation vector.
                inner = orient_line[1:-1].strip()
                if ',' in inner:
                    parts = [p.strip() for p in inner.split(',') if p.strip()]
                else:
                    parts = inner.split()
                if len(parts) == 3:
                    try:
                        x, y, z = map(float, parts)
                    except ValueError:
                        continue
                else:
                    continue
                # Parse rsm line: use the last element as the CD value.
                parts_rsm = rsm_line.split()
                try:
                    cd_value = float(parts_rsm[-1])
                except (ValueError, IndexError):
                    cd_value = 0.0
                
                xs.append(x)
                ys.append(y)
                zs.append(z)
                cds.append(cd_value)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(xs, ys, zs, c=cds, cmap='coolwarm', marker='o', s=60)
        print(np.std(cds))
        ax.set_title(f'{species} Orientation Vectors Colored by CD Value (rsm)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        plt.colorbar(sc, ax=ax, label='CD Value')
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process orientation vectors and extract CD values.')
    parser.add_argument('--print', action='store_true', help='Print each orientation with its corresponding CD value')
    parser.add_argument('--plot', action='store_true', help='Plot each orientation endpoint in a 3D scatter plot, colored by CD value')
    parser.add_argument('--blocks', type=int, nargs='+', help='Specify block numbers to process (e.g., --blocks 1 2)')
    parser.add_argument('--species', type=str, choices=['N2', 'O'], default='N2',
                        help='Select species for the results directory (default: N2)')
    parser.add_argument('--plotmode', type=str, choices=['sparta', 'rsm'], default='sparta',
                        help='Select plot mode: "sparta" uses drag_force files, "rsm" uses rsm_n2_results.dat (default: sparta)')
    args = parser.parse_args()
    
    blocks_to_use = args.blocks if args.blocks else None

    if args.print:
        print_orientations_with_cd(species=args.species, blocks_to_use=blocks_to_use)
    if args.plot:
        plot_orientations_3d(species=args.species, blocks_to_use=blocks_to_use, plotmode=args.plotmode)