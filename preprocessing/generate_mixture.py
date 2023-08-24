import math
import os
import numpy as np
import random
import csv
import scipy.io
from scipy import signal
from tqdm import tqdm
import wfdb
from utils import *

# Define a function to get file paths with stimulus information (STI)
def get_filepaths_withSTI(directory, ftype='.npy'):
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Check if the file is not stimulus info ('i') and matches the specified file type
            if filename[-5] != 'i' and filename.endswith(ftype):
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)  # Add the file path to the list
    return sorted(file_paths)  # Return the sorted list of file paths

# Define a function to add noise to a clean signal
def add_noise(clean_path, noise_path, SNR, return_info=False, normalize=False):
    clean_rate = 1000
    y_clean = np.load(clean_path)
    noise_ori = np.load(noise_path)
    
    #if noise shorter than clean wav, extend
    if len(noise_ori) < len(y_clean):
        tmp = (len(y_clean) // len(noise_ori)) + 1
        y_noise = []
        for _ in range(tmp):
            y_noise.extend(noise_ori)
    else:
        y_noise = noise_ori

    # cut noise 
    start = random.randint(0,len(y_noise)-len(y_clean))
    end = start+len(y_clean)
    y_noise = y_noise[start:end]     
    y_noise = np.asarray(y_noise)

    y_clean_pw = np.dot(y_clean,y_clean) 
    y_noise_pw = np.dot(y_noise,y_noise) 

    scalar = np.sqrt(y_clean_pw/((10.0**(SNR/10.0))*y_noise_pw))
    noise = scalar * y_noise
    y_noisy = y_clean + noise
    if normalize: 
        norm_scalar = np.max(abs(y_noisy))
        y_noisy = y_noisy/norm_scalar

    if return_info is False:
        return y_noisy, clean_rate
    else:
        info = {}
        info['start'] = start
        info['end'] = end
        info['scalar'] = scalar
        return y_noisy, clean_rate, info
    
# Save annotation data to CSV files for each split
def save_annotations_to_csv(annotations, split_name, exercise):
    if annotations:  # Check if the annotations list is not empty
        csv_filename = f'../{split_name}_annotations_E{exercise}.csv'
        with open(csv_filename, 'w', newline='') as csv_file:
            fieldnames = annotations[0].keys()
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(annotations)
        print(f"Annotations saved for {split_name} dataset.")
    else:
        print(f"No annotations to save for {split_name} dataset.")

# Define parameters for train, validation and test sets
noise_paths = ["../ECG_Ch1_fs1000_bp_training", "../ECG_Ch1_fs1000_bp_validation", "../ECG_Ch2_fs1000_bp_testing"]
exercise = 1
channel = [1, 2, 3, 4, 5, 6, 9, 10]

# values = [round(value, 2) for value in np.arange(start_value, end_value + step_size, step_size)]
SNR_train = [round(snr, 2) for snr in np.arange(0, -16, -1)]
SNR_val = [round(snr, 2) for snr in np.arange(0, -15.5, -0.5)]
SNR_test = [round(snr, 2) for snr in np.arange(0, -15.5, -0.5)]
SNR_lists = [SNR_train, SNR_val, SNR_test]
# num_of_copy = [1, 1, 1]
normalize = True  # Specify if the output noisy EMG should be normalized
sti = False  # Specify if the data includes stimulus information

# Create lists to hold annotation data for each split
train_annotations = []
val_annotations = []
test_annotations = []

# Loop through each specified channel
for ch in channel:
    emg_folder = f"../data_E{exercise}_S40_Ch{ch}_withSTI_seg10s_nsrd"
    clean_paths = [emg_folder + "/train/clean", emg_folder + "/val/clean", emg_folder + "/test/clean"]
    out_path = f"../mixed_signals_E{exercise}"
    check_path(out_path)
    # Loop through each clean path
    for i in range(len(clean_paths)):
        clean_path = clean_paths[i]
        noise_path = noise_paths[i]
        SNR_list = SNR_lists[i]
        if i == 2:
            sti = True
        # Get lists of clean EMG and noise file paths
        clean_list = get_filepaths(clean_path) if sti is False else get_filepaths_withSTI(clean_path)
        noise_list = get_filepaths(noise_path)
        
        sorted(clean_list)  # Sort the list of clean EMG paths
        
        # Loop through SNR values
        for snr in SNR_list:
            # Loop through each clean EMG path
            for clean_emg_path in tqdm(clean_list):
                # Randomly select noise files
                noise_wav_path_list = random.sample(noise_list, 1)
                # Loop through noise files and create noisy EMG signals
                for noise_ecg_path in noise_wav_path_list:
                    y_noisy, clean_rate, info = add_noise(clean_emg_path, noise_ecg_path, snr, True, normalize)
                    emg_name = clean_emg_path.split(os.sep)[-1].split(".")[0]
                    noise_name = noise_ecg_path.split(os.sep)[-1].split(".")[0]
                    mixed_name = emg_name + "_" + noise_name + "_" + str(snr)
                    np.save(os.path.join(out_path, mixed_name), y_noisy)  # Save noisy EMG signal
                    
                    # Create a dictionary to hold annotation data
                    annotation_data = {
                        'mixed_name': mixed_name + ".npy",
                        'snr': snr,
                        'emg_filename': emg_name,
                        'ecg_filename': noise_name
                        # Add other annotations if needed
                    }
                    # Append the annotation data to the appropriate split's list
                    if i == 0:
                        train_annotations.append(annotation_data)
                    elif i == 1:
                        val_annotations.append(annotation_data)
                    elif i == 2:
                        test_annotations.append(annotation_data)    

# Save annotation data for each split
save_annotations_to_csv(train_annotations, 'train', exercise)
save_annotations_to_csv(val_annotations, 'val', exercise)
save_annotations_to_csv(test_annotations, 'test', exercise)