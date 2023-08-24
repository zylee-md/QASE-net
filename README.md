# Deep Learning-based Signal-to-Noise Ratio Estimation for Single-Channel Surface EMG Signals Contaminated by ECG Interference

# Dataset
1. ECG: [MIT-BIH Normal Sinus Rhythm Database](https://www.physionet.org/content/nsrdb/1.0.0/) 
2. sEMG: [NINAPro database DB2](http://ninaweb.hevs.ch/node/17)

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
