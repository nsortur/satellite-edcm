import re
import os

samples_path = "/Users/neelsortur/Documents/codestuff/sat-modeling/satellite-edcm/data/sparta_dria"

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
    
# find any outliers with drag coefficient > 3
def find_outliers():
    # Get all files in the samples directory
    files = [f for f in os.listdir(samples_path) if f.endswith('.txt')]
    
    outlier_files = []
    
    # Read data from all files
    for file in files:
        file_path = os.path.join(samples_path, file)
        data = read(file_path)
        
        if 'drag_coeff' in data:
            # Parse drag coefficient
            drag_coeff = float(data['drag_coeff'])
            
            if drag_coeff > 3.0:
                outlier_files.append((file, drag_coeff))
    
    return outlier_files

if __name__ == "__main__":
    outliers = find_outliers()
    if outliers:
        print("Outlier files with drag coefficient > 3.0:")
        for file, drag_coeff in outliers:
            print(f"{file}: {drag_coeff}")
    else:
        print("No outliers found with drag coefficient > 3.0.")
    