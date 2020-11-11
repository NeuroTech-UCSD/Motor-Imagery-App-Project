import pandas as pd
import numpy as np
from neurodsp import filt
import random
import matplotlib.pyplot as plt
import scipy.signal as signal # For filtering
from neurodsp.spectral import compute_spectrum # for smoothed PSD computation
import pyeeg


# Filter eeg // does perform well on short signals, possibly because of padding
def filterEEG(eeg_data, fs=250, f_range=(1, 50)):
    sig_filt = filt.filter_signal(eeg_data, fs, 'bandpass', f_range, filter_type='iir', butterworth_order=2)
    test_sig_filt = filt.filter_signal(sig_filt, fs, 'bandstop', (58, 62), n_seconds=1)
    num_nans = sum(np.isnan(test_sig_filt))
    sig_filt = np.concatenate(([0]*(num_nans // 2), sig_filt, [0]*(num_nans // 2)))
    sig_filt = filt.filter_signal(sig_filt, fs, 'bandstop', (58, 62), n_seconds=1)
    sig_filt = sig_filt[~np.isnan(sig_filt)]
    return sig_filt

from scipy.signal import butter, sosfiltfilt, sosfreqz  # for filtering
def bandpass_bandstop_filter(data,fs=250, lowcut=1, highcut=50, order = 2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog = False, btype = 'band', output = 'sos')
    filted_data = sosfiltfilt(sos, data)
    filted_data = filt.filter_signal(filted_data, fs, 'bandstop', (58, 62), n_seconds=1)
    filted_data = filted_data[~np.isnan(filted_data)]
    return filted_data

# Filter a whole set of eeg epochs 
def getFilteredEpochs(eeg_epochs):
    # eeg_epochs is in (#trials, #timepoints, #channels)
    filtered_eeg_epochs = []
    for eeg_epoch in eeg_epochs: 
        filtered_epoch = []
        for i in range(len(eeg_epoch)):  
            filtered_eeg = bandpass_bandstop_filter(eeg_epoch[i])
            filtered_epoch.append(filtered_eeg)
        filtered_epoch = np.array(filtered_epoch)
        filtered_eeg_epochs.append(filtered_epoch)
    return np.array(filtered_eeg_epochs)

def getOutputLabelsAndEpochTimes(event_df):
    # Generates the ordered list of output labels and epoch time pairs
    # Input: event_df -- the event dataframe from the csv file
    # Output: 
    #    output_labels -- [1, 2, 4, 3, etc] where the integers correspond to the trial type encoded
    #    epoch_times -- [[<timestamp of start>, <timestamp of end>], [<timestamp of start>, <timestamp of end>], etc]
    
    output_labels = []
    epoch_times = []
    current_epoch = []
    for index, row in event_df.iterrows():
        event_info = row['EventStart'].split("_")
        if event_info[0] == 'start': 
            output_labels.append(int(event_info[1]))
            current_epoch.append(row['time'])
        else :
            current_epoch.append(row['time'])
            epoch_times.append(list(current_epoch))
            current_epoch = []
    return np.array(output_labels), np.array(epoch_times)
def getEEGEpochs(epoch_times, eeg_df, eeg_chans, target_num_trials=1000):
    # Slices and generates the epochs in the eeg_df given the epoch_times
    # Input: 
    #    epoch_times: [[<timestamp of start>, <timestamp of end>], [<timestamp of start>, <timestamp of end>], etc]
    #    eeg_df: dataframe from csv file
    # Output: 
    #    a numpy array containing eeg_epochs (#epoch, #chans, #timepoints)
    eeg_epochs = []
    for epoch_time in epoch_times: 
        baseline_df = eeg_df[(eeg_df['time'] > (epoch_time[0] - 1.5)) & (eeg_df['time'] < ((epoch_time[0])))]
        baseline_df = baseline_df.drop(columns=['time'])
        baselines = np.mean(baseline_df[eeg_chans].values, 0)
        sub_df = eeg_df[(eeg_df['time'] > epoch_time[0]) & (eeg_df['time'] < epoch_time[1])]
        sub_df = sub_df.drop(columns=['time'])
        sub_df = sub_df - baselines
        num_above = len(sub_df) - target_num_trials
        if num_above >= 0:
            epoch = np.array(sub_df.values[num_above // 2: len(sub_df) - num_above // 2])[:1000]
            eeg_epochs.append(epoch.T)
            if len(epoch) != 1000:
                print("Warning: Potential off by 1 error. Found trail with != 1000 samples:", len(epoch))
        else: 
            print("Warning: Epoch with less than", target_num_trials, "eeg samples")
    return np.array(eeg_epochs)




## Create DF for each of these, columns are channels, each row is a trial run
def getDF(epochs, labels, times, chans):
    data_dict = {}
    for i, label in enumerate(labels): 
        start_time = times[i][0]
        if 'start_time' not in data_dict: 
            data_dict['start_time'] = list()
        data_dict['start_time'].append(start_time)
        
        if 'event_type' not in data_dict:
            data_dict['event_type'] = list()
        data_dict['event_type'].append(label)
        
        for ch in range(len(chans)): 
            if chans[ch] not in data_dict:
                data_dict[chans[ch]] = list() 
            data_dict[chans[ch]].append(epochs[i][ch])
        
    return pd.DataFrame(data_dict)

# PSD plotting
def plotPSD(freq, psd, fs=250, pre_cut_off_freq=0, post_cut_off_freq=120, label=None):
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('log(PSD)')
    pre_cut = int(len(freq)*(pre_cut_off_freq / freq[-1]))
    post_cut = int(len(freq)*(post_cut_off_freq / freq[-1]))
    plt.plot(freq[pre_cut:post_cut], np.log(psd[pre_cut:post_cut]), label=label)

def getFreqPSDFromEEG(eeg_data, fs=250):
    freq, psd = signal.periodogram(eeg_data, fs=int(fs), scaling='spectrum')
    return freq, psd

def getMeanFreqPSD(eeg_data, fs=250):
    freq_mean, psd_mean = compute_spectrum(eeg_data, fs, method='welch', avg_type='mean', nperseg=fs*2)
    return freq_mean, psd_mean

# Plot PSD from EEG data 
def plotPSD_fromEEG(eeg_data, fs=250, pre_cut_off_freq=0, post_cut_off_freq=120, label=None):
    freq, psd = signal.periodogram(eeg_data, fs=int(fs), scaling='spectrum')
    plotPSD(freq, psd, fs, pre_cut_off_freq, post_cut_off_freq, label)

# Get Powerbin information from EEG data
def getPowerRatio(eeg_data, binning, eeg_fs=250):
    power, power_ratio = pyeeg.bin_power(eeg_data, binning, eeg_fs)
    return np.array(power_ratio)
def getIntervals(binning): 
    intervals = list()
    for i, val in enumerate(binning[:-1]): 
        intervals.append((val, binning[i+1]))
    return intervals