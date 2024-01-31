import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from models import WLDNN  # Adjust the model import as per your saved model
import argparse
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import gc

# Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2024, help='Seed used during testing')
parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use')
args = parser.parse_args()

# Set device
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
print("Device set to", device)

# Load the trained model
model_path = f'./checkpoints_wldnn/wldnn_30eps_seed{args.seed}.pth'  # Adjust the path and filename as per your saving convention
model = WLDNN(50).to(device)
model_name = model.__class__.__name__
model.load_state_dict(torch.load(model_path))
model.eval()

# Function to calculate waveform length
def calculate_waveform_length(signal, window_size=200, overlap=0):
    # Calculate the number of segments based on segment size and overlap.
    signal_length = len(signal)
    step_size = int(window_size * (1 - overlap))
    num_segments = (signal_length - window_size) // step_size + 1
    segment_waveform_lengths = []
    for i in range(num_segments):
        start = i * step_size
        end = start + window_size
        segment = signal[start:end]
        # Calculate the waveform length for the current segment and store it in the result array.
        segment_length = np.sum(np.abs(np.diff(segment)))
        segment_waveform_lengths.append(segment_length)
    segment_waveform_lengths = np.array(segment_waveform_lengths)
    return segment_waveform_lengths

# Function to normalize a time series to have mean 0 and standard deviation 1
def mean_variance_normalize(time_series, mean=None, std=None):
    if mean is None:
        mean = np.mean(time_series)
    if std is None:
        std = np.std(time_series)
    
    # Ensure there's no division by zero.
    if std == 0:
        raise ValueError("Cannot normalize: Standard deviation is zero.")
    
    # Normalize the time series to have mean 0 and standard deviation 1.
    normalized_series = (time_series - mean) / std
    
    return normalized_series

# Function to normalize a signal to have unit energy
def normalize_to_unit_energy(signal):
    energy = np.sum(np.abs(signal)**2)
    
    # Ensure there's no division by zero.
    if energy == 0:
        raise ValueError("Cannot normalize: Energy is zero.")
    
    # Normalize the signal to have unit energy by dividing by the square root of the energy.
    normalized_signal = signal / np.sqrt(energy)
    
    return normalized_signal

# Function to load data
def load_data(df, data_folder):
    data, labels = [], []
    for _, row in df.iterrows():
        file_name, snr = row['mixed_name'], row['snr']
        signal = np.load(os.path.join(data_folder, file_name))
        
        # Apply mean variance normalization and energy normalization during loading
        signal = mean_variance_normalize(signal)
        signal = normalize_to_unit_energy(signal)
        
        wl = calculate_waveform_length(signal)
        data.append(wl)
        labels.append([snr])
    
    # Load normalization statistics based on the seed used during testing
    stats_file_path = f'./train_stats_{model_name.lower()}.txt'
    with open(stats_file_path, 'r') as file:
        lines = file.readlines()
        train_mean = float(lines[0].split(':')[-1])
        train_std = float(lines[1].split(':')[-1])

    # Normalize test data after feature extraction
    data, labels = np.array(data), np.array(labels)
    data = normalize_dataset(data, train_mean, train_std)
    
    return data, labels

# Function to normalize dataset by subtracting mean and dividing by standard deviation
def normalize_dataset(data, mean, std):
    normalized_data = [(item - mean) / std for item in data]
    return np.array(normalized_data)

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
