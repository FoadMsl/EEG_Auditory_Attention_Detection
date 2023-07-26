#=================================================
#   ML_Project__Auditory Attention Detection (on a part of KULeuven Dataset)
#   1_3_EEG_Processing
#   Foad Moslem - PhD Student - Aerodynamics
#   Using Python 3.11.4 & Spyder IDE
#=================================================

#%%
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass


# Libraries
import numpy as np
import pandas as pd
import mne # Import the MNE library for processing and analyzing EEG data
import sklearn
import sklearn.preprocessing


#%%
# Read EEG files name
OurEEG_Names = pd.read_csv("./1_3_EEG_10Ch_csv_Files/OurEEG_Names.csv")

# A for loop for applying what we want on EEG files one by one
for i in range(0,OurEEG_Names.shape[0]):
# for i in range(0,1): #for_test
    
    epochs_array = []
    
    ### Load The EEG Files =====================================
    # Call EEG files one by one
    EEG = OurEEG_Names.iloc[i][0]
    # Read the EEG file
    EEG_data = pd.read_csv(f"./1_3_EEG_10Ch_csv_Files/{EEG}.csv")
    # ==========================================================
    
    ### Create the info structure needed by MNE ================
    # The channel names
    ch_names = list(EEG_data.columns)
    # The channel types
    ch_types = ['eeg'] * EEG_data.columns.shape[0]
    # Sample Frequencies
    sfreq = 128 # Original sample rate (Hz)
    # Info
    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    # ==========================================================
    
    ### Create the EEG Raw object ==============================
    EEG_raw = mne.io.RawArray(EEG_data.T, info) # transpose eeg_data to match channels x samples
    # ==========================================================
    
    ### Downsample The EEG Files ===============================
    re_sfreq = 64 # Resampled rate (Hz)
    EEG_raw.resample(sfreq=re_sfreq) # downsample by a factor of 4
    # ==========================================================
    
    ### Filter The EEG Files ===================================
    EEG_raw.filter(l_freq=1, h_freq=31.999) # band-pass filter between 1 Hz and 32 Hz
    # ==========================================================
    
    ### Normalize Filtered EEG Signals =========================
    # Get the numpy array from the raw object
    EEG_raw_data = EEG_raw.get_data()
    # Normalize by channel
    EEG_raw_data_scaled = sklearn.preprocessing.scale(EEG_raw_data, axis=1)
    # Info2
    info2 = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=re_sfreq)
    # Create a new raw object with the scaled data
    EEG_raw_scaled = mne.io.RawArray(EEG_raw_data_scaled, info2)
    # ==========================================================
    
    ### Segment The EEG File Into epochs =======================
    # Create events every 3 seconds with 2 seconds overlap
    events = mne.make_fixed_length_events(EEG_raw_scaled, duration=3, overlap=2)
    # Create epochs from events with 3 seconds duration
    epochs = mne.Epochs(EEG_raw_scaled, events, event_id=1, tmin=0, tmax=3, baseline=None)
    # ==========================================================
    
    ### Raw Signal =============================================
    # Convert epochs into a suitable format for the model
    # Get raw EEG data for each epoch
    epochs_raw_data = epochs.get_data()
    # ==========================================================
    
    ### Save the Raw Signal to a csv file ======================
    RawSignals = np.array(epochs_raw_data)
    np.save(f"./1_4_EEG_10Ch_RawSignal/RawSignal_{EEG}.npy", RawSignals)
    # ==========================================================
