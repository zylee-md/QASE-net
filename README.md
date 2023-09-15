# QASE-net: A Signal-to-Noise Ratio Prediction Model for Single-Channel Surface EMG Signals Contaminated by ECG Interference

# Open database
1. sEMG (signal): [NINAPro database DB2](http://ninaweb.hevs.ch/node/17)
2. ECG (noise): [MIT-BIH Normal Sinus Rhythm Database](https://www.physionet.org/content/nsrdb/1.0.0/) 


# Directory Structure
```
/SNR-Estimation-for-sEMG-Signals
├── data_E{i}_S{j}_Ch{k}_withSTI_seg{SEGMENT_SIZE}s_nsrd
│   ...
├── ECG_Ch{ch}_fs1000_bp_training
├── ECG_Ch{ch}_fs1000_bp_testing
├── mixed_signals_E{i}
│   ├── mixed_signal_1.npy
│   │   ...
│   ├── mixed_signal_N.npy
├── {phase}_annotations_E{i}.csv
│   ...
└── preprocessing
    ├── extract_ecg.py
    ├── extract_emg.py
    ├── utils.py
    └── generate_mixture.py
```  

# Usage
* Download the dataset from the link provided above
* Run `extract_emg.py`, `extract_ecg.py`, and `generate_mixture.py` to generate training, testing, and validation data.
