import torch
from models import InvariantDragMLP
from groups import GSpaceInfo
from dataset.datasets import DragDataset
from torch_geometric.data import DataLoader
from tqdm import tqdm
import numpy as np

# Define the model arguments and corresponding weight paths
model_args = ['trivial', 'c2', 'c2c2', 'c2c2c2']
model_weights = {
    'trivial': '/Users/neelsortur/Documents/codestuff/sat-modeling/satellite-edcm/2024-12-16/15-55-47/0/weights/model_epoch_99.pt',
    'c2': '/Users/neelsortur/Documents/codestuff/sat-modeling/satellite-edcm/2024-12-16/16-16-47/0/weights/model_epoch_99.pt',
    'c2c2': '/Users/neelsortur/Documents/codestuff/sat-modeling/satellite-edcm/2024-12-16/16-13-24/0/weights/model_epoch_99.pt',
    'c2c2c2': '/Users/neelsortur/Documents/codestuff/sat-modeling/satellite-edcm/2024-12-16/15-54-07/0/weights/model_epoch_99.pt'
}

# Function to load a model with given arguments and weights
def load_model(arg, weight_path):
    model = InvariantDragMLP(GSpaceInfo(arg), hidden_dim=256)
    model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    model.eval()
    return model

# Function to perform metrics on the model
def perform_metrics(model, data_loader):
    total_eq_error = 0
    eq_errors = []
    num_samples = len(data_loader)
    sample_count = int(0.01 * num_samples)

    pbar = tqdm(enumerate(data_loader), total=sample_count)
    for i, (sample, y) in pbar:
        if i >= sample_count:
            break

        output_initial = model(sample)
        eq_error = 0
        for element in c2c2c2_elements:
            action = c2c2c2_rep(element)

            dir_vec = sample[:, 0:3].reshape((sample.shape[0], 3))
            dir_vec_rotated = dir_vec @ action

            sample_rotated = torch.cat((dir_vec_rotated.float(), sample[:, 3:].float()), axis=1)
            output_rotated = model(sample_rotated)
            eq_error += torch.nn.functional.mse_loss(output_initial, output_rotated).item()

        eq_error /= len(c2c2c2_elements)
        eq_errors.append(eq_error)
        total_eq_error += eq_error

    mean_eq_error = total_eq_error / sample_count
    std_eq_error = torch.tensor(eq_errors).std().item()

    return mean_eq_error, std_eq_error


ds = DragDataset("data/cube50k.dat", True)
data_loader = DataLoader(ds, batch_size=1, shuffle=False)

# Dictionary to store results
results = {}

c2c2c2 = GSpaceInfo('c2c2c2')
c2c2c2_elements = list(c2c2c2.group.fibergroup.testing_elements())
c2c2c2_rep = c2c2c2.input_reps[0]

# Iterate over each model argument, load the model, and perform metrics
for arg in model_args:
    with torch.no_grad():
        model = load_model(arg, model_weights[arg])
        metric_result = perform_metrics(model, data_loader)
        results[arg] = metric_result

# Show the results
for arg, result in results.items():
    print(f'Model with argument {arg}: Metric result = {result}')

import matplotlib.pyplot as plt

# Extract data for plotting
model_names = list(results.keys())
mean_errors = [results[arg][0] for arg in model_names]
std_errors = [results[arg][1] for arg in model_names]

# Create the bar plot
x_pos = np.arange(len(model_names))

fig, ax = plt.subplots()
ax.bar(x_pos, mean_errors, yerr=std_errors, align='center', alpha=0.7, capsize=10)
ax.set_yscale('log')
ax.set_xlabel('MLP Symmetry')
ax.set_ylabel('Average Equivariance Error (log scale)')
# ax.set_title('Equivariance Error by MLP Symmetry')
ax.set_xticks(x_pos)
ax.set_xticklabels(list(["Unconstrained", "$C_2$", "$C_2 \\times C_2$", "$C_2 \\times C_2 \\times C_2$"]))
ax.yaxis.grid(True)

# Save and show the plot
plt.tight_layout()
plt.savefig('equivariance_error_plot.png')
plt.show()