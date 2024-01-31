import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from models import *
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import argparse
import gc  # Import the garbage collection module

# Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2024, help='Random seed value')
parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use')
parser.add_argument('--epoch', type=int, default=30, help='Number of epochs')

args = parser.parse_args()

LR = 1e-3

# Set device to GPU if available, else CPU
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
print("Device set to", device)

# Paths to annotation CSV files
train_csv_path = "train_annotations_E1.csv"
val_csv_path = "val_annotations_E1.csv"

# Folder containing .npy files
data_folder = "./mixed_signals_E1"

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

set_random_seed(args.seed)

WINDOW_SIZE = 200

def calculate_waveform_length_segment(segment):
    # Calculate the waveform length as the sum of the absolute differences between adjacent samples in the segment.
    return np.sum(np.abs(np.diff(segment)))

def calculate_waveform_length(signal, window_size=WINDOW_SIZE, overlap=0):
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
        segment_length = calculate_waveform_length_segment(segment)
        segment_waveform_lengths.append(segment_length)
    segment_waveform_lengths = np.array(segment_waveform_lengths)
    return segment_waveform_lengths

def mean_variance_normalize(time_series):
    # Calculate the mean and standard deviation of the time series.
    mean = np.mean(time_series)
    std_dev = np.std(time_series)
    
    # Ensure there's no division by zero.
    if std_dev == 0:
        raise ValueError("Cannot normalize: Standard deviation is zero.")
    
    # Normalize the time series to have mean 0 and standard deviation 1.
    normalized_series = (time_series - mean) / std_dev
    
    return normalized_series

def normalize_to_unit_energy(signal):
    # Calculate the energy of the signal as the sum of the squares of its samples.
    energy = np.sum(np.abs(signal)**2)
    
    # Ensure there's no division by zero.
    if energy == 0:
        raise ValueError("Cannot normalize: Energy is zero.")
    
    # Normalize the signal to have unit energy by dividing by the square root of the energy.
    normalized_signal = signal / np.sqrt(energy)
    
    return normalized_signal

def calculate_overall_mean_std(array):
    # Calculate the mean and standard deviation of the input array.
    overall_mean = np.mean(array)
    overall_std = np.std(array)
    
    # Print the calculated values for reference.
    print(f"train_mean = {overall_mean}")
    print(f"train_std = {overall_std}")
    
    return overall_mean, overall_std

def normalize_dataset(data, mean, std):
    # Normalize the dataset by subtracting the mean and dividing by the standard deviation.
    normalized_data = (data - mean) / std
    return normalized_data

# Example usage:
test_sig = np.random.rand(10000,)
# print(test_sig.shape)
test_wl = calculate_waveform_length(test_sig)
# print(test_wl.shape)
input_length = test_wl.shape[0]

model = WLDNN(input_length).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# print(model)
# print("Number of trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
model_name = model.__class__.__name__

# Read the CSV files into pandas DataFrames
train_df = pd.read_csv(train_csv_path)
val_df = pd.read_csv(val_csv_path)

# Function to load data
def load_data(df, data_folder):
    data, labels = [], []
    for _, row in df.iterrows():
        file_name, snr = row['mixed_name'], row['snr']
        signal = np.load(os.path.join(data_folder, file_name))
        signal = mean_variance_normalize(signal)
        signal = normalize_to_unit_energy(signal)
        wl = calculate_waveform_length(signal)
        data.append(wl)
        labels.append(np.array([snr]))
    return np.array(data), np.array(labels)

# Load training and validation data
train_data, train_labels = load_data(train_df, data_folder)
val_data, val_labels = load_data(val_df, data_folder)

# Normalize data
train_mean, train_std = calculate_overall_mean_std(train_data)
train_data = normalize_dataset(train_data, train_mean, train_std)
val_data = normalize_dataset(val_data, train_mean, train_std)

wldnn_stats_dir = "./"
if not os.path.exists(wldnn_stats_dir):
    os.makedirs(wldnn_stats_dir)

# Save train_mean and train_std to a file
stats_file_path = os.path.join(wldnn_stats_dir, f'train_stats_{model_name.lower()}.txt')
with open(stats_file_path, 'w') as file:
    file.write(f"train_mean: {train_mean}\n")
    file.write(f"train_std: {train_std}\n")
print(f"Training statistics saved to {stats_file_path}")

# Convert to tensors and transfer to GPU
train_data_tensor = torch.Tensor(train_data).to(device)
train_labels_tensor = torch.Tensor(train_labels).to(device)
val_data_tensor = torch.Tensor(val_data).to(device)
val_labels_tensor = torch.Tensor(val_labels).to(device)

# Free up CPU memory
del train_data, train_labels, val_data, val_labels
gc.collect()

# Create DataLoaders from the tensors
batch_size = 32
# print("Creating DataLoaders...")

train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# print("Train data shape:", train_data_tensor.shape)
# print("Train labels shape:", train_labels_tensor.shape)

val_dataset = TensorDataset(val_data_tensor, val_labels_tensor)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# print("Validation data shape:", val_data_tensor.shape)
# print("Validation labels shape:", val_labels_tensor.shape)

num_epochs = args.epoch
steps_per_print = 1000

# Initialize a list to store the losses
train_losses = []
val_losses = []  # Initialize a list to store validation losses

print("==========================Training...==========================")
for epoch in range(num_epochs):
    model.train()
    step_counter = 0  # Initialize step counter for the epoch
    epoch_loss = 0.0  # Initialize the epoch loss
    for batch_data, batch_labels in train_dataloader:
        optimizer.zero_grad()
        predictions = model(batch_data)
        loss = criterion(predictions, batch_labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()  # Accumulate the loss for the epoch

        # Increment step counter
        step_counter += 1

        # # Print loss every `steps_per_print` steps
        # if step_counter % steps_per_print == 0:
        #     print(f"Epoch [{epoch+1}/{num_epochs}], Step [{step_counter}/{len(train_dataloader)}], Loss: {loss.item():.4f}")

    # Calculate and record the mean loss for the epoch
    mean_epoch_loss = epoch_loss / len(train_dataloader)
    train_losses.append(mean_epoch_loss)

    # Validation loop
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0

    with torch.no_grad():
        for val_data, val_labels in val_dataloader:
            val_predictions = model(val_data)
            val_loss += criterion(val_predictions, val_labels).item()

    mean_val_loss = val_loss / len(val_dataloader)
    val_losses.append(mean_val_loss)

    # print(f"Epoch [{epoch+1}/{num_epochs}], Mean Epoch Loss: {mean_epoch_loss:.4f}, Mean Validation Loss: {mean_val_loss:.4f}")
    # print("==================================================")

# print("Training completed")
ckpt_dir = f'./checkpoints_{model_name.lower()}'
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
torch.save(model.state_dict(), f'./checkpoints_wldnn/{model_name.lower()}_{num_epochs}eps_seed{args.seed}.pth')
print(f"Model saved at ./checkpoints_wldnn/{model_name.lower()}_{num_epochs}eps_seed{args.seed}.pth")

# Plot and save the training and validation loss curves
epochs = range(1, num_epochs + 1)  # Epochs starting from 1 to num_epochs
plt.figure()
plt.plot(epochs, train_losses, label='Training Loss')
plt.plot(epochs, val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Create results directory if it doesn't exist
results_dir = f'./train_loss_curve_{model_name.lower()}'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Save the plot
plt.savefig(os.path.join(results_dir, f'loss_curve_{model.__class__.__name__.lower()}_{num_epochs}eps_seed{args.seed}.png'))
plt.close()
