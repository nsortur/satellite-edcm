import matplotlib.pyplot as plt
import re
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# group = gspaces.no_base_space(escnn.group.octa_group())
# input_reps = 1 * [group.fibergroup.standard_representation]

# samp = group.fibergroup.sample()
# print(input_reps[0](samp))

# plot all the 200 samples

samples_path = "/Users/neelsortur/Documents/codestuff/sat-modeling/satellite-edcm/data/sparta_dria_h_1123"
outlier_samples = [
    # "drag_force_Output_Processed181.txt",
    # "drag_force_Output_Processed175.txt",
    # "drag_force_Output_Processed173.txt"
]

def read(file_path):
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
                data[key] = match.group(1)
            else:
                print(f"No match found for {key}")
                print(f"Pattern: {pattern}")
                print(f"Content snippet: {content[:500]}")
        
        return data
    
    
def plot_drag_coefficients():
    # Get all files in the samples directory
    files = [f for f in os.listdir(samples_path) if f.endswith('.txt')]
    
    orientations = []
    drag_coeffs = []
    
    # Read data from all files
    for file in files:
        file_path = os.path.join(samples_path, file)
        data = read(file_path)
        if file in outlier_samples:
            print("Skipping outlier sample:", file)
            continue
        
        if 'orientation' in data and 'drag_coeff' in data:
            # Parse orientation as 3D vector
            orientation_str = data['orientation'].strip()
            orientation = [float(x.strip()) for x in orientation_str.split(',')]
            
            # Parse drag coefficient
            drag_coeff = float(data['drag_coeff'])
            
            orientations.append(orientation)
            drag_coeffs.append(drag_coeff)
    
    # Convert to numpy arrays for easier handling
    orientations = np.array(orientations)
    drag_coeffs = np.array(drag_coeffs)
    
    # Normalize drag coefficients to create scaling factors
    min_drag = np.min(drag_coeffs)
    max_drag = np.max(drag_coeffs)
    
    # Create scaling factors: map drag coefficients to range [0.3, 1.7]
    # This ensures points don't get too close to origin or too far away
    scale_factors = 0.3 + 1.4 * (drag_coeffs - min_drag) / (max_drag - min_drag)
    
    # Scale orientations by drag coefficient
    scaled_orientations = orientations * scale_factors.reshape(-1, 1)
    
    # Create 3D scatter plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create scatter plot with color mapping
    scatter = ax.scatter(scaled_orientations[:, 0], scaled_orientations[:, 1], scaled_orientations[:, 2], 
                        c=drag_coeffs, cmap='viridis', s=130, alpha=1)
    
    # Add colorbar
    colorbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
    colorbar.set_label('Drag Coefficient', fontsize=12)
    
    # Set labels and title
    ax.set_xlabel('Scaled Orientation X', fontsize=12)
    ax.set_ylabel('Scaled Orientation Y', fontsize=12)
    ax.set_zlabel('Scaled Orientation Z', fontsize=12)
    ax.set_title('Drag Coefficient vs Scaled Orientation\n(Point distance from origin represents drag magnitude)', fontsize=14)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add a small sphere at origin for reference
    ax.scatter([0], [0], [0], c='red', s=100, alpha=0.8, marker='o', label='Origin')
    ax.legend()
    
    # set bounds
    max_range = np.max(np.abs(scaled_orientations)) * 1.1
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range]) 
    
    plt.tight_layout()
    plt.show()
    
    print(f"Plotted {len(drag_coeffs)} data points")
    print(f"Drag coefficient range: {min_drag:.4f} to {max_drag:.4f}")
    print(f"Scaling factor range: {np.min(scale_factors):.2f} to {np.max(scale_factors):.2f}")

# Call the function to create the plot
plot_drag_coefficients()