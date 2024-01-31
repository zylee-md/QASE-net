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
args = parser.parse_args()

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

# Read the CSV files into pandas DataFrames
train_df = pd.read_csv(train_csv_path)
val_df = pd.read_csv(val_csv_path)

# Function to load data
def load_data(df, data_folder):
    data, labels = [], []
    for _, row in df.iterrows():
        file_name, snr = row['mixed_name'], row['snr']
        data.append(np.load(os.path.join(data_folder, file_name)).reshape(1, -1))
        labels.append(np.array([snr]))
    return np.array(data), np.array(labels)

# Load training and validation data
train_data, train_labels = load_data(train_df, data_folder)
val_data, val_labels = load_data(val_df, data_folder)

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
print("Train data shape:", train_data_tensor.shape)
print("Train labels shape:", train_labels_tensor.shape)

val_dataset = TensorDataset(val_data_tensor, val_labels_tensor)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
print("Validation data shape:", val_data_tensor.shape)
print("Validation labels shape:", val_labels_tensor.shape)

model = CNNBLSTMATTN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

print(model)
print("Number of trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
model_name = model.__class__.__name__

num_epochs = 20
steps_per_print = 400

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

        # Print loss every `steps_per_print` steps
        if step_counter % steps_per_print == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{step_counter}/{len(train_dataloader)}], Loss: {loss.item():.4f}")

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

    print(f"Epoch [{epoch+1}/{num_epochs}], Mean Epoch Loss: {mean_epoch_loss:.4f}, Mean Validation Loss: {mean_val_loss:.4f}")
    print("==================================================")

# print("Training completed")
ckpt_dir = f'./checkpoints_{model_name.lower()}'
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
torch.save(model.state_dict(), f'./checkpoints_{model_name.lower()}/{model_name.lower()}_{num_epochs}eps_seed{args.seed}.pth')
print(f"Model saved at checkpoints_{model_name.lower()}/{model_name.lower()}_{num_epochs}eps_seed{args.seed}.pth")

# Plot and save the training and validation loss curves
epochs = range(1, num_epochs + 1) 
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
