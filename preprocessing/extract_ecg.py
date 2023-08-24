import os
import wfdb
import numpy as np
from scipy import signal as sig
from tqdm import tqdm
from utils import check_path, resample

# Function to get ECG file paths from a directory
def get_ecg_filepaths(directory):
    # Train  3M + 9F
    train_subject_ids = [16265, 16272, 16273, 16420, 16483, 16539, 16773, 16786, 16795, 17052, 17453, 18177]
    train_ecg_paths = [os.path.join(directory, str(subject)) for subject in train_subject_ids]
    # Test 2M + 4F
    test_subject_ids = [18184, 19088, 19090, 19093, 19140, 19830]
    test_ecg_paths = [os.path.join(directory, str(subject)) for subject in test_subject_ids]
    return train_ecg_paths, test_ecg_paths

# Function to read and preprocess ECG signal
def read_ecg(ecg_path, channel=1):
    ecg_record = wfdb.rdrecord(ecg_path)
    ecg_signal = ecg_record.p_signal[:, channel-1]
    ecg_rate = 1000
    ecg_resampled = resample(ecg_signal, 128, ecg_rate)
    ecg_resampled = ecg_resampled.astype('float64')
    return ecg_resampled

# Function to extract and save filtered ECG signals
def extract_ecg(file_paths, output_path, start, end, b_h, a_h, b_l, a_l, channel=1):
    # Create the output directory if it doesn't exist
    check_path(output_path)

    # Iterate through ECG files
    for ecg_path in tqdm(file_paths):
        # Read and preprocess the ECG signal
        ecg_signal = read_ecg(ecg_path, channel)
        file_name = os.path.split(ecg_path)[-1].split(".")[0]

        # Apply filters and save the filtered signals
        if ecg_signal.ndim > 1:
            for j in range(ecg_signal.shape[1]):
                filtered_ecg = sig.filtfilt(b_l, a_l, sig.filtfilt(b_h, a_h, ecg_signal[start:end, j]))
                np.save(os.path.join(output_path, f"{file_name}{j}"), filtered_ecg)
        else:
            filtered_ecg = sig.filtfilt(b_l, a_l, sig.filtfilt(b_h, a_h, ecg_signal[start:end]))
            np.save(os.path.join(output_path, file_name), filtered_ecg)

# Set up paths and filter parameters
corpus_path = '../../mit-bih-normal-sinus-rhythm-database-1.0.0'
train_out_path = '../ECG_Ch1_fs1000_bp_training'
val_out_path = '../ECG_Ch1_fs1000_bp_validation'
test_out_path = '../ECG_Ch2_fs1000_bp_testing'

# High-pass and low-pass filter parameters
b_h, a_h = sig.butter(3, 10, 'hp', fs=1000)
b_l, a_l = sig.butter(3, 200, 'lp', fs=1000)

# Segment period
start, end = 0, 70000            

# Get ECG file paths
train_ecg_paths, test_ecg_paths = get_ecg_filepaths(corpus_path)

# Extract and save filtered ECG signals
extract_ecg(train_ecg_paths, train_out_path, start, end, b_h, a_h, b_l, a_l, channel=1)
extract_ecg(test_ecg_paths, val_out_path, start, end, b_h, a_h, b_l, a_l, channel=1)
extract_ecg(test_ecg_paths, test_out_path, start, end, b_h, a_h, b_l, a_l, channel=2)