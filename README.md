# QASE-net: A Signal-to-Noise Ratio Prediction Model for Single-Channel Surface EMG Signals Contaminated by ECG Interference
In practical scenarios involving the measurement of surface electromyography (sEMG) in muscles, particularly those areas near the heart, one of the primary sources of contamination is the presence of electrocardiogram (ECG) signals. To effectively and efficiently quantify the quality of real-world sEMG data, this study proposes a novel non-intrusive quality assessment model, termed QASE-net, to predict the SNR of sEMG signals. QASE-net employs a combination of CNN-BLSTM with attention mechanisms and follows an end-to-end training strategy. Notably, our experimental framework utilizes authentic sEMG and ECG data open-access databases, the Non-Invasive Adaptive Prosthetics database and the MIT-BIH Normal Sinus Rhythm Database, respectively. The experimental results demonstrate the superiority of QASE-net over the previous assessment model, exhibiting significantly reduced prediction errors and notably higher linear correlations with the ground truth. These findings demonstrate the potential of QASE-net to substantially enhance the reliability and precision of sEMG quality assessment in practical applications.

# Open database
1. sEMG (signal): [NINAPro database DB2](http://ninaweb.hevs.ch/node/17)
2. ECG (noise): [MIT-BIH Normal Sinus Rhythm Database](https://www.physionet.org/content/nsrdb/1.0.0/) 

# Directory Structure
```
/EMG_DB2 (the EMG dataset)
/mit-bih-normal-sinus-rhythm-database-1.0.0 (the ECG dataset)
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

# Data Preparation
* Download the dataset from the link provided above
* Run `extract_emg.py` and `extract_ecg.py` to extract sEMG and ECG signals from the downloaded database
* To generated contaminated sEMG signals, run `generate_mixture.py`

# Model Training and Testing
* Run `train-{model_name}.ipynb` to train the models and `test-{model_name}.ipynb` to test them
