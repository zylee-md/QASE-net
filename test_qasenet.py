import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from models import *
import argparse
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import gc

# Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2024, help='Seed used during training')
parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use')
args = parser.parse_args()

# Set device
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
print("Device set to", device)

# Load the trained model
model_path = f'./checkpoints_cnnblstmattn/cnnblstmattn_20eps_seed{args.seed}.pth'  # Adjust the path and filename as per your saving convention
model = CNNBLSTMATTN().to(device)
model_name = model.__class__.__name__
model.load_state_dict(torch.load(model_path))
model.eval()

# Function to load data
def load_data(df, data_folder):
    data, labels = [], []
    for _, row in df.iterrows():
        file_name, snr = row['mixed_name'], row['snr']
        data.append(np.load(os.path.join(data_folder, file_name)).reshape(1, -1))
        labels.append([snr])
    return np.array(data), np.array(labels)

# Load test data
test_csv_path = "./test_annotations_E1.csv"
data_folder = "./mixed_signals_E1"
test_df = pd.read_csv(test_csv_path)
test_data, test_labels = load_data(test_df, data_folder)

# Convert to tensors and DataLoader
test_dataset = TensorDataset(torch.Tensor(test_data).to(device), torch.Tensor(test_labels).to(device))
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Free up CPU memory
del test_data, test_labels
gc.collect()

# Test the model and collect predictions
predictions = []
targets = []
with torch.no_grad():
    for data, labels in test_dataloader:
        output = model(data)  # Unpack the tuple
        predictions.extend(output.cpu().numpy())
        targets.extend(labels.cpu().numpy())
        
# Convert to numpy arrays for calculation
predictions = np.array(predictions)
targets = np.array(targets)

# Concatenate predictions and labels
predictions_np = np.concatenate(predictions, axis=0)
targets_np = np.concatenate(targets, axis=0)

# Reshape predictions and labels
predictions_flat = predictions_np.reshape(-1)
targets_flat = targets_np.reshape(-1)

# Function to calculate metrics
def calculate_metrics(true, pred):
    mse = mean_squared_error(true, pred)
    mae = mean_absolute_error(true, pred)
    lcc = pearsonr(true, pred)[0]
    srcc = spearmanr(true, pred)[0]
    return mse, mae, lcc, srcc

# Calculate metrics for SNR
mse_snr, mae_snr, lcc_snr, srcc_snr = calculate_metrics(targets_flat, predictions_flat)

results_dir = f"./test_results_{model_name}_20eps"

# Save metrics to a file
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

metrics_file = os.path.join(results_dir, f'test_metrics_{model_name}_seed{args.seed}.txt')
with open(metrics_file, 'w') as f:
    f.write(f'SNR Metrics:\n')
    f.write(f'MSE:{mse_snr}\n')
    f.write(f'MAE:{mae_snr}\n')
    f.write(f'LinearCC:{lcc_snr}\n')
    f.write(f'SpearmanCC:{srcc_snr}\n\n')

print(f"Testing completed. Metrics saved to {metrics_file}.")
