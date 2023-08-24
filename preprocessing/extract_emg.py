import numpy as np
import os
import scipy.io
from scipy import signal
from utils import *
from tqdm import tqdm

# Function to generate EMG file paths
def get_emg_filepaths(directory, number, exercise):
    emg_paths = []
    for i in range(1, number + 1):
        filename = f"DB2_s{i}/S{i}_E{exercise}_A1.mat"
        emg_paths.append(os.path.join(directory, filename))
    return emg_paths

# Function to read and process EMG data from a given .mat file
def read_emg(emg_path, channel, restimulus=False):
    b, a = signal.butter(4, [20, 500], 'bp', fs=2000)
    emg_data = scipy.io.loadmat(emg_path)
    y_clean = emg_data.get('emg')[:, channel - 1]
    y_clean = signal.filtfilt(b, a, y_clean)[::2]
    y_clean = y_clean / np.max(np.abs(y_clean))
    y_clean = y_clean.astype('float64')
    if restimulus:
        y_restimulus = emg_data.get('restimulus')[::2]
    else:
        y_restimulus = 0
    return y_clean, y_restimulus

# Define paths and parameters
Corpus_path = '../../EMG_DB2/'
segment = 10  # length in seconds
points_per_seg = segment * 1000
EMG_data_num = 40

# Channels and corresponding phases to process
channels_to_process = [
    # channel, split, exercise
    [1, "train", 1],
    [2, "train", 1],
    [3, "train", 1],
    [4, "train", 1],
    [5, "val", 1],
    [6, "val", 1],
    [9, "test", 1],
    [10, "test", 1]
]

# Loop through each channel-phase-exercise combination
for ch, phase, exercise in channels_to_process:
    # Construct the output path
    out_path = f"../data_E{exercise}_S{EMG_data_num}_Ch{ch}_withSTI_seg{segment}s_nsrd"
    
    # Ensure required directories exist
    check_path(out_path)
    check_path(f"{out_path}/train/clean")
    check_path(f"{out_path}/val/clean")
    check_path(f"{out_path}/test/clean")

    # Get EMG file paths
    file_paths = get_emg_filepaths(Corpus_path, EMG_data_num, exercise)

    # Determine the file range based on phase
    if phase == "train":
        start_idx = 0
        end_idx = 24
    elif phase == "val":
        start_idx = 24
        end_idx = 32
    else:  # phase == "test"
        start_idx = 32
        end_idx = len(file_paths)

    # Loop through each EMG file
    for i in tqdm(range(start_idx, end_idx)):
        # Determine if the current file is for testing
        test = phase == "test"

        # Construct the save path based on the phase
        save_path = f"{out_path}/{phase}/clean"

        # Read and process EMG data
        emg_file, restimulus = read_emg(file_paths[i], ch, test)
        
        # Segment and save EMG data
        for j in range(emg_file.shape[0] // points_per_seg):
            np.save(
                os.path.join(save_path, f"{file_paths[i].split(os.sep)[-1].split('.')[0]}_ch{ch}_{j}"),
                emg_file[j * points_per_seg:(j + 1) * points_per_seg]
            )
            if test:
                np.save(
                    os.path.join(save_path, f"{file_paths[i].split(os.sep)[-1].split('.')[0]}_ch{ch}_{j}_sti"),
                    restimulus[j * points_per_seg:(j + 1) * points_per_seg]
                )
