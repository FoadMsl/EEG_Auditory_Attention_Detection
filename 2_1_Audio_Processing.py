#=================================================
#   ML_Project__Auditory Attention Detection (on a part of KULeuven Dataset)
#   2_1_Audio_Processing
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
import pandas as pd
import librosa
import scipy
import matplotlib.pyplot as plt
import numpy as np
import scipy.io


#%% 
# Read Audio files name
OurAudio_Names = pd.read_csv("./2_2_Audio_dry_wav_Files/OurAudio_Names.csv")

# A for loop for applying what we want on audio files one by one
for i in range(0,OurAudio_Names.shape[0]):
# for i in range(0,2): #for_test
    
    spectrograms = []

    ### Load The Audio Files =====================================
    # call audio files one by one
    audio = OurAudio_Names.iloc[i][0]
    # make the audio file path for further actions
    audio_data = (f"./2_2_Audio_dry_wav_Files/{audio}")
    # Load the audio file
    y , sr = librosa.load(audio_data, sr=None)
    # ============================================================
        
    ### Filter The Audio Files ===================================
    # Design a low pass filter with a cut-off frequency of 8 kHz
    sos = scipy.signal.butter(10, 8000, btype="lowpass", output="sos", fs=sr)    
    # Apply the filter to the audio signal
    y_filtered = scipy.signal.sosfilt(sos, y)
    # ============================================================
    
    ### Downsample The Audio Files ===============================
    # Define the target sampling rate based on reference paper
    target_sr=16000
    
    # Downsample the filtered signal to 16 kHz
    y_downsampled = librosa.resample(y_filtered, orig_sr=sr, target_sr=target_sr)
    # ============================================================

    ### Segment The Audio File Into Trials =======================
    # Define the duration and overlap of the trials in seconds
    duration = 3
    overlap = 2
    # Compute the number of samples for each trial
    samples_per_trial = int(duration * target_sr)
    # Compute the hop length between trials
    hop_length_trial = int((duration - overlap) * target_sr)
    # Segment the audio into trials
    trials = librosa.util.frame(y_downsampled, frame_length=samples_per_trial, hop_length=hop_length_trial)
    # ============================================================
        
    ### Short-Time Fourier Transform (STFT) & Spectrograms =======
    # Define the window length and overlap for STFT in samples
    # Convert 32 ms and 12 ms to samples by multiplying with the sampling rate
    n_fft = int(0.032 * target_sr)
    win_length = n_fft
    hop_length_stft = int(0.012 * target_sr)
    
    
    # Loop over the trials and compute the STFT and spectrogram for each one
    for j in range(trials.shape[1]):
    # for j in range(0,2): #for_test
        print(f"{audio}_trial{j+1}")
        
        trial = trials[:, j]

        ### Spectrogram ==============================================
        # Compute the STFT of the trial using a Hann window
        D = librosa.stft(trial, n_fft=n_fft, win_length=win_length, hop_length=hop_length_stft, window="hann")
        Xdb = librosa.amplitude_to_db(abs(D))
        
        # Compute the magnitude spectrogram of the trial by taking the absolute value of the STFT coefficients
        S = np.abs(Xdb)
        spectrograms.append(S)
        # ============================================================
                
        ### Plot the spectrogram using a logarithmic frequency scale==
        # fig1 = plt.figure(figsize=(10, 5), dpi=300, clear=False, layout="constrained")
        # plt.xlabel("Time")
        # plt.ylabel("dB")
        # plt.title(f"Spectrogram of {audio}_trial{j+1}")
        
        # # librosa.display.specshow(S, sr=target_sr, x_axis='time', y_axis='hz')
        # # convert the frequency axis to a logarithmic one
        # librosa.display.specshow(S, sr=target_sr, x_axis='time', y_axis='log')
        # plt.colorbar(format='%+2.0f dB')
        
        # # Save The Spectrogram Images
        # plt.savefig(f"./Audio_Dataset_Spectrogram/Spectrogram_of_trial{j+1}_{audio}.png")
        
        # # Close the figure to avoid displaying it
        # plt.close(fig1)
        # # Show the figure without blocking
        # plt.show(block=False)
        # # Clear the current figure
        # plt.clf()
        # ============================================================
    
    # save the spectrograms to a csv file
    Spec = np.array(spectrograms)
    np.save(f"./2_3_Audio_dry_Spectrogram/spectrogram_{audio}.npy", Spec)

#%%
# Spec2 = np.load(f"./Audio_Dataset_Spectrogram/spectrogram_part4_track2_dry.wav.npy")
#%% Explanations:
"""       
    #=================================================
    scipy.signal.butter(N, Wn, btype='low', analog=False, output='ba', fs=None)

    The butter function in scipy.signal.butter is used to design a Butterworth
    filter and return the filter coefficients. A Butterworth filter is a type 
    of filter that has a maximally flat frequency response in the passband.
        
    N: the order of the filter, which determines how steep the transition from 
        passband to stopband is.
    Wn: the critical frequency or frequencies, which define the edges of the 
        passband. For lowpass and highpass filters, Wn is a scalar.
    btype: the type of filter, which can be ‘lowpass’, ‘highpass’, ‘bandpass’,
        or ‘bandstop’.
    analog: a boolean flag that indicates whether to return an analog filter 
        or a digital filter.
    output: the type of output, which can be ‘ba’ (numerator and denominator 
        polynomials), ‘zpk’ (zeros, poles, and system gain), or ‘sos’ 
        (second-order sections).
    fs: the sampling frequency of the digital system.

    #=================================================
    scipy.signal.sosfilt(sos, x, axis=-1, zi=None)
    
    The scipy.signal.sosfilt function is used to filter data along one 
    dimension using cascaded second-order sections.
    sos: an array of second-order filter coefficients, with shape 
        (n_sections, 6). Each row corresponds to a second-order section, 
        with the first three columns providing the numerator coefficients and 
        the last three providing the denominator coefficients.
    x: an N-dimensional input array of data to be filtered.
    axis: the axis of the input data array along which to apply the filter. 
        The default is -1, which means the last axis.
    zi: an optional array of initial conditions for the filter delays. It has 
        shape (n_sections, …, 2, …), where …, 2, … denotes the shape of x, but 
        with x.shape [axis] replaced by 2. If not given, initial rest (i.e. all
        zeros) is assumed.
        
    #=================================================
    librosa.resample(y, *, orig_sr, target_sr, res_type='soxr_hq', fix=True, 
                     scale=False, axis=-1, **kwargs)
    
    Resample a time series from orig_sr to target_sr
    
    y: audio time series, with n samples along the specified axis.
    orig_sr: original sampling rate of y
    target_sr: target sampling rate
    res_type: resample type        
    
    #=================================================
    librosa.util.frame(x, *, frame_length, hop_length, axis=-1, writeable=False, 
                       subok=False)
    
    Slice a data array into (overlapping) frames.
    
    x: Array to frame
    frame_length: Length of the frame
    hop_length: Number of steps to advance between frames
    axis: The axis along which to frame.
    writeable: If True, then the framed view of x is read-only. If False, then 
        the framed view is read-write. Note that writing to the framed view will 
        also write to the input array x in this case.
    subok: If True, sub-classes will be passed-through, otherwise the returned 
        array will be forced to be a base-class array (default).
        
    #=================================================
        librosa.stft(y, *, n_fft=2048, hop_length=None, win_length=None, 
                      window='hann', center=True, dtype=None, pad_mode='constant', 
                      out=None)
        
        The STFT represents a signal in the time-frequency domain by computing 
        discrete Fourier transforms (DFT) over short overlapping windows.
        
        y: input signal. Multi-channel is supported.
        n_fft: length of the windowed signal after padding with zeros.
        hop_length: number of audio samples between adjacent STFT columns.
        win_length: Each frame of audio is windowed by window of length 
            win_length and then padded with zeros to match n_fft.
        window: ...      
        
    #=================================================
        
    """

