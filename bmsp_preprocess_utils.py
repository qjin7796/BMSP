# General utility functions:
#     read_d1namo_csv, fill_missing_values, correct_outliers, resample_signal
# ECG signal preprocessing functions:
#     denoise_ecg, locate_r_peaks, compute_heart_rate, compute_hrv, assess_ecg_quality,
#     extract_qrs, delineate_qrs, extract_ecg_epochs, group_ecg_epochs
# Accelerometer signal preprocessing functions:
#     denoise_acc, compute_vector_magnitude, extract_acc_epochs
# Multi-modal signal preprocessing functions:
#     align_acc_to_ecg
# ------------------
# Author: Qiuhan Jin
# Date: 2024-08-16


import os, datetime, pickle, pandas as pd, numpy as np
import neurokit2 as nk, matplotlib.pyplot as plt, plotly as py


#######################
### General Utility ###
#######################
def read_d1namo_csv(file_path: str, timestamp_colname: str) -> pd.DataFrame | str | int:
    ''' Read (and encode the timestamp column to asceding numbers,) with valid datetime
        and return signal time-series, start time, sampling rate, sampling duration.
    '''
    signal_df = pd.read_csv(file_path, sep=',', header=0)
    # Check if the timestamp column exists
    if timestamp_colname not in signal_df.columns:
        raise ValueError(f'Timestamp column {timestamp_colname} not found!')
    else:
        # Convert to pd.Timestamp
        signal_df[timestamp_colname] = pd.to_datetime(
            signal_df[timestamp_colname], errors='coerce'
        )
        # Detect invalid values, including NaT, np.nan, np.inf, empty string ''
        invalid_indices = signal_df[timestamp_colname].isnull()
        num_invalid = invalid_indices.sum()
        if num_invalid > 0:
            raise ValueError(f'{num_invalid} invalid timestamps detected!')
        else:
            start_timestamp = signal_df[timestamp_colname][0]
            next_timestamp = signal_df[timestamp_colname][1]
            # Write start_timestamp as YYYYMMDD_HHMMSS_US
            reformat_timestamp = lambda x: (
                f'{x.year:04d}{x.month:02d}{x.day:02d}_'
                f'{x.hour:02d}{x.minute:02d}{x.second:02d}_{x.microsecond:06d}'
            )
            start_time_str = reformat_timestamp(start_timestamp)
            sampling_rate = 1000000 // (next_timestamp.microsecond - start_timestamp.microsecond)  # Hz
            num_samples = signal_df.shape[0]
            sampling_duration = num_samples / (sampling_rate * 60)  # minutes
            return signal_df, start_time_str, sampling_rate, sampling_duration

def fill_missing_values(signal: pd.Series) -> pd.Series | int:
    ''' Fill missing values, None/np.nan, with linear interpolation. '''
    filled_signal = signal.copy()
    num_missing_values = signal.isna().sum()
    if num_missing_values > 0:
        filled_signal.interpolate(inplace=True, limit_direction='both')
    return filled_signal, num_missing_values

def correct_outliers(signal: pd.Series, exclude=2, side='both') -> pd.Series | int:
    ''' Correct outliers, based on standard deviation, with linear interpolation. 
        exclude * std(signal) as the threshold.
        side: {'both', 'left', 'right'}.
    '''
    corrected_signal = signal.copy()
    # Detect outliers: boolean array
    outliers = nk.find_outliers(corrected_signal, exclude=exclude, side=side, method='sd')
    num_outliers = outliers.sum()
    # Replace outliers with interpolation
    corrected_signal[outliers] = np.nan
    corrected_signal.interpolate(inplace=True, limit_direction='both')
    return corrected_signal, num_outliers

def resample_signal(signal: pd.Series, sampling_rate: int, target_rate: int) -> pd.Series:
    ''' Resample signal to match the target sampling rate. '''
    resampled_signal = nk.signal_resample(
        signal, sampling_rate=sampling_rate, 
        desired_sampling_rate=target_rate, method='interpolation'
    )
    return resampled_signal


#############################
### Preprocess ECG Signal ###
#############################
def denoise_ecg(signal: pd.Series, sampling_rate: int, method='neurokit') -> pd.Series:
    ''' Remove baseline wander and powerline interference from the signal. 
        method: {'neurokit', 'biosppy'}
            'neurokit' method: 
                A 0.5 Hz high-pass butterworth filter (order = 5) + 50 Hz powerline filter.
            'biosppy' method: 
                A [0.67, 45] Hz FIR filter (order = 1.5 * SR).
        Signal is recentered before filtering.
    '''
    denoised_signal = signal.copy()
    denoised_signal[:] = nk.ecg_clean(
        signal, method=method, sampling_rate=sampling_rate
    )  # ecg_colname
    return denoised_signal

def locate_r_peaks(signal: pd.Series, sampling_rate: int, method='rodrigues2021') -> pd.Series | np.ndarray:
    ''' Locate R peaks in the ECG signal. 
        method: {'rodrigues2021', 'promac'}
            'rodrigues2021' method:
                A low computational cost, adaptive algorithm for fast R-peak detection.
            'promac' method: https://github.com/neuropsychology/NeuroKit/issues/222
                A probabilistic approach that estimates the prob distribution of R-peaks 
                by ensemble learning of several methods' predictions.
        sampling_rate: int, Hz to resample the signal.
        Return 
            a boolean series of the same length, R peaks encoded in 1 (single timepoint),
            a np.ndarray of the index of R peaks.
    '''
    r_peaks, _ = nk.ecg_peaks(signal, sampling_rate=sampling_rate, method=method)  # 'ECG_R_Peaks'
    r_peaks_indices = np.array(r_peaks[r_peaks['ECG_R_Peaks'] == 1].index)  # or _['ECG_R_Peaks']
    return r_peaks, r_peaks_indices

def compute_heart_rate(r_peaks: np.ndarray, sampling_rate: int) -> np.ndarray:
    ''' Compute heart rate of detected R-peaks. 
        Returns a np.float64 array, heart rate in bpm.
    '''
    heart_rate = nk.ecg_rate(r_peaks, sampling_rate=sampling_rate)
    return heart_rate

def compute_hrv(r_peaks: np.ndarray, sampling_rate: int) -> np.ndarray:
    ''' Estimate heart rate variability from R peaks.
        Return a dictionary of HRV indices.
    '''
    hrv_indices = nk.hrv(r_peaks, sampling_rate=sampling_rate)
    return hrv_indices

def assess_ecg_quality(signal: pd.Series, sampling_rate: int, rpeaks: np.ndarray) -> pd.Series:
    ''' Assess signal quality based on the detected R-peaks. 
        method: 'averageQRS'
            Computes the distance of a QRS segment to the average QRS segment in the signal.
            This distance [0, 1] is a relative index of similarity with the signal average.
        Return a float series of the same length, signal quality index [0, 1], 1 is best match.
    '''
    quality_array = nk.ecg_quality(
        signal, rpeaks=rpeaks, sampling_rate=sampling_rate, method='averageQRS'
    )  # np.float64 array
    signal_quality_indices = pd.Series(quality_array, index=signal.index, name='ECG_SQA')  # 'ECG_SQA'
    return signal_quality_indices

def extract_qrs(signal: pd.Series, sampling_rate: int, rpeaks: np.ndarray) -> dict:
    ''' Segment ECG signal into QRS complexes. 
        Return a dictionary of the QRS epoch for every R peak.
        { index: { 'Index', 'Signal', 'R_Peaks' } }
    '''
    qrs_epochs = nk.ecg_segment(signal, rpeaks=rpeaks, sampling_rate=sampling_rate)
    return qrs_epochs

def delineate_qrs(signal: pd.Series, sampling_rate: int, rpeaks: np.ndarray) -> pd.Series | dict:
    ''' Delineate P waves, Q waves, ST segments, T waves from ECG signal. 
        Return 
            a bool series of the same length, onsets/offsets/peaks encoded in 1, 
            a dictionary of the delineated waves for every QRS complex,
            { ECG_P_Onsets, ECG_P_Peaks, ECG_P_Offsets, 
              ECG_Q_Peaks, ECG_R_Onsets, ECG_R_Offsets, ECG_S_Peaks, 
              ECG_T_Onsets, ECG_T_Peaks, ECG_T_Offsets }.
    '''
    qrs_waves_indices, qrs_waves = nk.ecg_delineate(
        signal, rpeaks=rpeaks, sampling_rate=sampling_rate
    )
    return qrs_waves_indices, qrs_waves

def extract_ecg_epochs(signal: pd.Series, sampling_rate: int, rpeaks: np.ndarray,
                       sqa_indices: pd.Series, sqa_threshold=0.8) -> dict | np.ndarray:
    ''' Divide signal into segments of single heart beats of good quality. 
        Return 
            a dictionary of all QRS epochs { ECG_R_Peaks, ECG_QRS_Epochs },
                ECG_QRS_Epochs: { epoch_index: { 'Index', 'Signal' } },
            a dictionary of good epochs { ECG_QRS_Labels, ECG_QRS_Epochs },
                ECG_QRS_Labels: np.ndarray of QRS indices (in int labels),
                ECG_QRS_Epochs: { epoch_label: { 'Index', 'Length', 'ECG_R_Peaks', ... } }.
    '''
    # Extract QRS epochs and waves of the entire cleaned signal
    all_qrs_epochs = extract_qrs(signal, sampling_rate, rpeaks)
    num_all_qrs_epochs = len(all_qrs_epochs)
    print(f'{datetime.datetime.now()}: {num_all_qrs_epochs} QRS epochs detected.')
    _, all_qrs_waves = delineate_qrs(signal, sampling_rate, rpeaks)
    all_qrs_epochs_info = {
        'ECG_R_Peaks': rpeaks,
        'ECG_QRS_Epochs': all_qrs_epochs,
    }
    for wave_name, wave_indices in all_qrs_waves.items():
        all_qrs_epochs_info[wave_name] = np.array(wave_indices)
    print(f'{datetime.datetime.now()}: {len(all_qrs_waves)} types of QRS waves delineated.')
    # Filter out epochs with signal quality index < sqa_threshold
    good_qrs_epochs_info = {
        'ECG_QRS_Labels': np.zeros(signal.shape[0], dtype=np.int64),
        'ECG_QRS_Epochs': {},
    }
    # Use ascending numbers to label selected epochs' indices in the signal
    epoch_label = 0
    for epoch_i, epoch_info in all_qrs_epochs.items():
        ## Skip first and last epoch because of padding
        if epoch_i == '1' or epoch_i == str(num_all_qrs_epochs):
            continue
        epoch_indices = epoch_info['Index'].tolist()
        ## Compute min signal quality index of the epoch
        epoch_sqa_min = sqa_indices[epoch_indices].min()
        if epoch_sqa_min >= sqa_threshold:
            epoch_label += 1
            good_qrs_epochs_info['ECG_QRS_Labels'][epoch_indices] = epoch_label
            ## Search epoch_indices in rpeaks using set intersection
            epoch_rpeaks = np.fromiter(set(epoch_indices) & set(rpeaks), dtype=np.int64)
            good_qrs_epochs_info['ECG_QRS_Epochs'][epoch_label] = {
                'Index': np.array(epoch_indices),
                'Length': len(epoch_indices),
                'ECG_R_Peaks': epoch_rpeaks,
            }
            for wave_name, wave_indices in all_qrs_waves.items():
                ## Find wave indices within the epoch_indices
                epoch_wave_indices = np.fromiter(set(epoch_indices) & set(wave_indices), dtype=np.int64)
                good_qrs_epochs_info['ECG_QRS_Epochs'][epoch_label][wave_name] = epoch_wave_indices
    return all_qrs_epochs_info, good_qrs_epochs_info

def group_ecg_epochs(epochs: dict, sampling_rate: int, group_window=180) -> dict:
    ''' Group consecutive epochs into segments, segment length >= group_window seconds.
        Return a dictionary of segments { segment: { indices, signal, rpeaks } }.
    '''
    # Concatenate 'index' of all epochs
    all_signal_indices = np.concatenate([v['index'] for v in epochs.values()])
    all_signal_values = np.concatenate([v['signal'] for v in epochs.values()])
    epoch_keys = np.concatenate([np.repeat(k, len(v['index'])) for k, v in epochs.items()])
    # Group epoch_signal_indices into segments if diff == 1 (consecutive indices)
    group_of_signal_indices = np.split(all_signal_indices, np.where(np.diff(all_signal_indices) > 1)[0] + 1)
    group_of_signal_values = np.split(all_signal_values, np.where(np.diff(all_signal_indices) > 1)[0] + 1)
    group_of_epoch_keys = np.split(epoch_keys, np.where(np.diff(all_signal_indices) > 1)[0] + 1)
    # Filter out segments with length < group_thr
    segment_dict = {}
    for i, group in enumerate(group_of_signal_indices):
        epoch_keys = np.unique(group_of_epoch_keys[i]).tolist()
        group_rpeaks = sorted([n for j in epoch_keys for n in epochs[j]['rpeaks']])
        if len(group_rpeaks) >= (group_window * sampling_rate):
            segment_dict[i] = {
                'indices': group,
                'epoch_keys': epoch_keys,
                'signal': group_of_signal_values[i],
                'rpeaks': np.array(group_rpeaks),
            }
    return segment_dict

def preproc_ecg(
        file_path: str, timestamp_colname: str, ecg_colname: str,
        outlier_exclude: int = 2, outlier_side: str = 'both',
        denoise_method: str = 'neurokit', quality_threshold: float = 0.8,
        group_epochs_window: int = 180, log_file: str = None, output_file: str = None
    ):
    ''' Preprocess ECG signal through the following steps:
        1. Clean the raw signal: 
            fill missing values, correct outliers, denoise the signal.
        2. Assess the signal quality: 
            locate R peaks, compute signal quality index.
        3. Compute heart rate and estimate HRV.
        4. Extract single heart beat epochs of good quality: 
            segment signal around every single R peaks,
            select the segments based on signal quality index,
            delineate each segment into QRS waves.
        Save a dictionary to a pickle file:
            dict keys: [ 'ecg_signal', 'ecg_clean_info', 'ecg_peaks_info' ].
            dict values:
                ecg_signal = cleaned ecg signal pd.Series, 
                ecg_clean_info = summary of raw and cleaned signal stats,
                ecg_peaks_info including good_r_peaks, good_epochs, good_qrs_waves etc.,
                    good_epochs = { epoch_index: { index, length, signal, rpeaks } },
                    good_qrs_waves = { ECG_P_Onsets: np.ndarray, ECG_P_Peaks, ECG_P_Offsets, 
                                       ECG_Q_Peaks, ECG_R_Onsets, ECG_R_Offsets, ECG_S_Peaks, 
                                       ECG_T_Onsets, ECG_T_Peaks, ECG_T_Offsets }.
    '''
    time_0 = datetime.datetime.now()
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(f'{datetime.datetime.now()} Preprocessing ECG signal file: {file_path}\n')
    # 1.1 Read the csv file
    ecg_df, start_time_str, ecg_res, ecg_dur = read_d1namo_csv(file_path, timestamp_colname)
    ecg_signal = ecg_df[ecg_colname].copy()
    # Compute raw signal stats
    ecg_len = ecg_signal.size
    ecg_raw_mean = ecg_signal.mean()
    ecg_raw_std = ecg_signal.std()
    bins = [ecg_raw_mean - 2*ecg_raw_std, ecg_raw_mean + 2*ecg_raw_std]
    ecg_raw_2std_pctg = ecg_signal.value_counts(bins=bins, sort=False).sum() / ecg_len
    ecg_raw_flat_pctg = nk.signal_flatline(ecg_signal, threshold=0.01)
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(
                f'Start time: {start_time_str}.\n'
                f'ECG signal colname: {ecg_colname}.\n'
                f'ECG sampling rate: {ecg_res} Hz, '
                f'duration: {ecg_len} samples = {ecg_dur} minutes.\n'
                f'ECG raw signal mean: {ecg_raw_mean}, std: {ecg_raw_std}.\n'
                f'ECG raw signal within 2*std range percentage: {ecg_raw_2std_pctg}.\n'
                f'ECG raw signal flatline percentage: {ecg_raw_flat_pctg}.\n\n'
            )
    # 1.2 Fill missing values
    ecg_signal, num_missing_values = fill_missing_values(ecg_signal)
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(f'{num_missing_values} missing values filled.\n')
    # 1.3 Correct outliers
    ecg_signal, num_outliers = correct_outliers(
        ecg_signal, exclude=outlier_exclude, side=outlier_side
    )
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(f'{num_outliers} outliers corrected.\n')
    # 1.4 Denoise the signal
    ecg_signal = denoise_ecg(
        ecg_signal, sampling_rate=ecg_res, method=denoise_method
    )
    # Compute cleaned signal stats
    ecg_cleaned_mean = ecg_signal.mean()
    ecg_cleaned_std = ecg_signal.std()
    bins = [ecg_cleaned_mean - 2*ecg_cleaned_std, ecg_cleaned_mean + 2*ecg_cleaned_std]
    ecg_cleaned_2std_pctg = ecg_signal.value_counts(bins=bins, sort=False).sum() / ecg_len
    ecg_cleaned_flat_pctg = nk.signal_flatline(ecg_signal, threshold=0.01)
    # Summarize signal cleaning
    time_1 = datetime.datetime.now()
    print(f'{time_1}: ECG signal cleaned.')
    ecg_clean_info = {
        'signal_colname': ecg_colname, 'timestamp_colname': timestamp_colname,
        'start_time': start_time_str, 'sampling_rate': ecg_res, 
        'duration': ecg_dur, 'num_samples': ecg_len,
        'raw_stats': {
            'mean': ecg_raw_mean, 'std': ecg_raw_std,
            '2std_pctg': ecg_raw_2std_pctg, 'flat_pctg': ecg_raw_flat_pctg
        },
        'num_missing_values': num_missing_values, 'num_outliers': num_outliers,
        'cleaned_stats': {
            'mean': ecg_cleaned_mean, 'std': ecg_cleaned_std,
            '2std_pctg': ecg_cleaned_2std_pctg, 'flat_pctg': ecg_cleaned_flat_pctg
        },
        'time_elapsed': (time_1 - time_0).total_seconds()
    }
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(
                f'ECG denoised with {denoise_method} method.\n'
                f'ECG denoised signal mean: {ecg_cleaned_mean}, std: {ecg_cleaned_std}.\n'
                f'ECG denoised signal within 2*std range percentage: {ecg_cleaned_2std_pctg}.\n'
                f'ECG denoised signal flatline percentage: {ecg_cleaned_flat_pctg}.\n'
                f'Cleaning time elapsed: {(time_1 - time_0).total_seconds()} seconds.\n\n'
            )
    # 2.1 Locate R peaks
    ecg_r_peaks, ecg_r_peaks_indices = locate_r_peaks(ecg_signal, ecg_res)
    time_2 = datetime.datetime.now()
    print(f'{time_2}: {len(ecg_r_peaks_indices)} ECG R peaks detected.')
    # 2.2 Assess signal quality
    ecg_sqa = assess_ecg_quality(ecg_signal, ecg_res, ecg_r_peaks_indices)
    time_3 = datetime.datetime.now()
    print(f'{time_3}: ECG signal quality assessed.')
    # 3 Compute heart rate and heart rate variability
    ecg_rate = compute_heart_rate(ecg_r_peaks, ecg_res)  # np.float64 array
    mean_ecg_rate = ecg_rate.mean()  # bpm
    # ecg_hrv = compute_hrv(ecg_r_peaks, ecg_res)  # dict
    time_4 = datetime.datetime.now()
    print(f'{time_4}: ECG heart rate computed.')
    # 4.1 Segment epochs (QRS complex) of good quality and delineate QRS waves
    ecg_all_epochs_info, ecg_good_epochs_info = extract_ecg_epochs(
        ecg_signal, ecg_res, ecg_r_peaks_indices, 
        ecg_sqa, quality_threshold
    )
    num_all_epochs = len(ecg_all_epochs_info['ECG_QRS_Epochs'])
    num_good_epochs = len(ecg_good_epochs_info['ECG_QRS_Epochs'])
    # # 4.2 Group consecutive epochs into segments for futher analysis
    # ecg_good_segments = group_ecg_epochs(ecg_good_epochs, ecg_res, group_epochs_window)
    # Summarize signal peaks properties
    time_5 = datetime.datetime.now()
    print(f'{time_5}: {num_good_epochs} ECG QRS epochs extracted.')
    ecg_peaks_info = {
        'r_peaks': ecg_r_peaks, 'r_peaks_indices': ecg_r_peaks_indices, 
        'num_r_peaks': len(ecg_r_peaks_indices), 'signal_quality': ecg_sqa,
        'heart_rate': ecg_rate, # 'heart_rate_variability': ecg_hrv,
        'num_all_epochs': num_all_epochs, 'all_epochs_info': ecg_all_epochs_info, 
        'num_good_epochs': num_good_epochs, 'good_epochs_info': ecg_good_epochs_info,
        'time_elapsed': (time_5 - time_1).total_seconds(),
        'locate_r_peaks_time': (time_2 - time_1).total_seconds(),
        'assess_ecg_quality_time': (time_3 - time_2).total_seconds(),
        'compute_heart_rate_time': (time_4 - time_3).total_seconds(),
        'extract_epochs_time': (time_5 - time_4).total_seconds()
    }
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(
                f'{len(ecg_r_peaks_indices)} R peaks detected.\n'
                f'Mean heart rate: {mean_ecg_rate:.1f} bpm.\n'
                f'Signal quality threshold: {quality_threshold}.\n'
                f'{num_good_epochs} out of {num_all_epochs} epochs extracted.\n'
                f'Segment analysis time elapsed: {(time_5 - time_1).total_seconds()} seconds.\n'
                f'    Locate R peaks time elapsed: {(time_2 - time_1).total_seconds()} seconds.\n'
                f'    Assess signal quality time elapsed: {(time_3 - time_2).total_seconds()} seconds.\n'
                f'    Compute heart rate time elapsed: {(time_4 - time_3).total_seconds()} seconds.\n'
                f'    Extract epochs time elapsed: {(time_5 - time_4).total_seconds()} seconds.\n\n'
            )
    if output_file is not None:
        with open(output_file, 'wb') as f:
            pickle.dump({
                'ecg_signal': ecg_signal, 
                'ecg_clean_info': ecg_clean_info, 
                'ecg_peaks_info': ecg_peaks_info
            }, f)
    return ecg_signal, ecg_clean_info, ecg_peaks_info

#######################################
### Preprocess Accelerometer Signal ###
#######################################
def denoise_acc(signal: pd.DataFrame, signal_colname_list: list, sampling_rate: int, 
                lowcut: float = None, highcut: float = None) -> pd.DataFrame:
    ''' Remove machine noise and powerline interference from the signal. 
        highcut/lowcut: cut-off frequency for a low-pass/high-pass filter.
            'ukbiobank' method: A 20 Hz low-pass butterworth filter (order = 4).
            'fridolfsson2019' method: A [0.29, 10 Hz] band-pass filter is recommended.
        Signal is recentered.
    '''
    denoised_signal = signal.copy()
    for col in signal_colname_list:
        denoised_signal[col] = nk.signal_filter(
            signal[col].astype(float), sampling_rate=sampling_rate, 
            lowcut=lowcut, highcut=highcut, method='butterworth', order=4
        )
    return denoised_signal

def compute_vector_magnitude(signal: pd.DataFrame, signal_colname_list: list) -> pd.DataFrame:
    ''' Compute the l2-norm (vector magnitude) of the multi-axis signal and its gradient.
        Append vector magnitude and its gradient to the signal dataframe.
    '''
    new_signal = signal.copy()
    signal_array = signal[signal_colname_list].to_numpy()
    vector_magnitude_array = np.linalg.norm(signal_array, axis=1)
    new_signal['Vector_Magnitude'] = vector_magnitude_array
    vector_magnitude_gradient_array = np.gradient(vector_magnitude_array)
    new_signal['Vector_Magnitude_Gradient'] = vector_magnitude_gradient_array
    return new_signal

def extract_acc_epochs(signal: pd.DataFrame, signal_colname_list: list, sampling_rate: int, 
                       duration: int = 5, overlap: int = 4) -> pd.Series | dict:
    ''' Extract epochs from the signal and compute stats for each epoch. 
        duration: duration of each epoch in seconds.
        overlap: adjacent epoch overlapping duration in seconds.
        Return
            a bool series of epoch centers,
            a dictionary of all epochs' & columns' stats,
            { colname: { mean: np.ndarray, std: np.ndarray } }.
    '''
    epoch_duration = duration * sampling_rate  # samples
    epoch_overlap = overlap * sampling_rate
    start_epoch_center = epoch_duration // 2
    end_epoch_center = signal.shape[0] - start_epoch_center
    epoch_indices = np.arange(start_epoch_center, end_epoch_center, epoch_duration - epoch_overlap)
    epoch_stats = {col: {
        'mean': np.zeros(len(epoch_indices)), 'std': np.zeros(len(epoch_indices))
    } for col in signal_colname_list}
    for i, center in enumerate(epoch_indices):
        epoch_start = center - epoch_duration // 2
        epoch_end = center + epoch_duration // 2
        for col in signal_colname_list:
            epoch_stats[col]['mean'][i] = signal[col][epoch_start:epoch_end].mean()
            epoch_stats[col]['std'][i] = signal[col][epoch_start:epoch_end].std()
    return epoch_indices, epoch_stats

def preproc_acc(file_path: str, timestamp_colname: str, signal_colname_list: list, 
                target_rate: int, lowcut: float = 0.29, highcut: float = 10., 
                window_size: int = 5, window_overlap: int = 4,
                log_file: str = None, output_file: str = None) -> dict:
    ''' Preprocess accelerometer signal through the following steps: 
        1. Clean the raw signal: 
            fill missing values, denoise the signal,
            resample the cleaned signal to the target_rate.
        2. Compute vector magnitude (l2-norm) of the 3-axis signal and gradient.
        3. Extract short, overlapping epochs and compute stats of the signal.
        Save a dictionary to a pickle file:
            dict keys: [ 'acc_signal', 'acc_clean_info', 'acc_epochs_info' ].
            dict values: 
                acc_signal = cleaned acc signal pd.DataFrame,
                acc_clean_info = summary of raw and cleaned signal stats,
                acc_epochs_info including epoch_stats,
                    epoch_stats: { epoch_index: { column: { mean, std } } }.
    '''
    time_0 = datetime.datetime.now()
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(f'{time_0}: Preprocessing accelerometer signal file: {file_path}\n')
    # 1.1 Read signal data
    acc_df, start_time_str, acc_res, acc_dur = read_d1namo_csv(file_path, timestamp_colname)
    acc_signal = acc_df[signal_colname_list].copy()
    # Compute raw signal stats
    acc_len = acc_signal.shape[0]
    acc_raw_mean = acc_signal[signal_colname_list].mean().to_numpy()
    acc_raw_std = acc_signal[signal_colname_list].std().to_numpy()
    acc_raw_2std_pctg = np.zeros(len(signal_colname_list), dtype=np.float64)
    acc_raw_flat_pctg = np.zeros(len(signal_colname_list), dtype=np.float64)
    for i, col in enumerate(signal_colname_list):
        bins = [acc_raw_mean[i] - 2*acc_raw_std[i], acc_raw_mean[i] + 2*acc_raw_std[i]]
        acc_raw_2std_pctg[i] = acc_signal[col].value_counts(bins=bins, sort=False).sum() / acc_len
        acc_raw_flat_pctg[i] = nk.signal_flatline(acc_signal[col], threshold=0.01)
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(
                f'Start time: {start_time_str}.\n'
                f'ACC signal columns: {signal_colname_list}.\n'
                f'ACC sampling rate: {acc_res} Hz, '
                f'duration: {acc_len} samples = {acc_dur} minutes.\n'
                f'ACC raw signal mean: {acc_raw_mean}, std: {acc_raw_std}.\n'
                f'ACC raw signal within 2*std range percentage: {acc_raw_2std_pctg}.\n'
                f'ACC raw signal flatline percentage: {acc_raw_flat_pctg}.\n\n'
            )
    # 1.2 Fill missing values
    acc_num_missing_values = np.zeros(len(signal_colname_list), dtype=np.int64)
    for i, col in enumerate(signal_colname_list):
        acc_signal[col], acc_num_missing_values[i] = fill_missing_values(acc_signal[col])
    # 1.3 Denoise signal
    acc_signal = denoise_acc(
        acc_signal, signal_colname_list, acc_res, lowcut=lowcut, highcut=highcut
    )
    # 1.4 Resample signal
    if target_rate is not None and target_rate != acc_res:
        resampled = pd.DataFrame(
            data={col: resample_signal(
                acc_signal[col], acc_res, target_rate
            ) for col in signal_colname_list},
            columns=signal_colname_list
        )
        acc_signal = resampled
    # 2 Compute vector magnitude
    acc_signal = compute_vector_magnitude(acc_signal, signal_colname_list)
    updated_signal_colname_list = acc_signal.columns.tolist()
    acc_len = acc_signal.shape[0]
    # Compute cleaned signal stats
    acc_clean_mean = acc_signal[updated_signal_colname_list].mean().to_numpy()
    acc_clean_std = acc_signal[updated_signal_colname_list].std().to_numpy()
    # acc_clean_25pct = acc_signal[updated_signal_colname_list].quantile(0.25).to_numpy()
    # acc_clean_75pct = acc_signal[updated_signal_colname_list].quantile(0.75).to_numpy()
    acc_clean_2std_pctg = np.zeros(len(updated_signal_colname_list), dtype=np.float64)
    acc_clean_flat_pctg = np.zeros(len(updated_signal_colname_list), dtype=np.float64)
    for i, col in enumerate(updated_signal_colname_list):
        bins = [acc_clean_mean[i] - 2*acc_clean_std[i], acc_clean_mean[i] + 2*acc_clean_std[i]]
        acc_clean_2std_pctg[i] = acc_signal[col].value_counts(bins=bins, sort=False).sum() / acc_len
        acc_clean_flat_pctg[i] = nk.signal_flatline(acc_signal[col], threshold=0.01)
    # Summarize signal cleaning
    time_1 = datetime.datetime.now()
    acc_clean_info = {
        'signal_colnames': updated_signal_colname_list, 'timstamp_colname': timestamp_colname,
        'start_time': start_time_str, 'sampling_rate': acc_res, 
        'duration': acc_dur, 'num_samples': acc_len,
        'raw_stats': {
            'mean': acc_raw_mean, 'std': acc_raw_std, 
            '2std_pctg': acc_raw_2std_pctg, 'flat_pctg': acc_raw_flat_pctg
        },
        'num_missing_values': acc_num_missing_values, 
        'lowcut': lowcut, 'highcut': highcut, 'resampled_rate': target_rate,
        'clean_stats': {
            'mean': acc_clean_mean, 'std': acc_clean_std, 
            '2std_pctg': acc_clean_2std_pctg, 'flat_pctg': acc_clean_flat_pctg
        },
        'time_elapsed': (time_1 - time_0).total_seconds()
    }
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(
                f'ACC denoised with lowcut: {lowcut} Hz, highcut: {highcut} Hz, '
                f'resampled to {target_rate} Hz.\n'
                f'ACC updated signal columns: {updated_signal_colname_list}.\n'
                f'ACC cleaned signal mean: {acc_clean_mean}, std: {acc_clean_std}.\n'
                f'ACC cleaned signal within 2*std range percentage: {acc_clean_2std_pctg}.\n'
                f'ACC cleaned signal flatline percentage: {acc_clean_flat_pctg}.\n'
                f'Cleaning time elapsed: {(time_1 - time_0).total_seconds()} seconds.\n\n'
            )
    # 3 Extract epochs and compute stats
    epoch_indices, epoch_stats = extract_acc_epochs(
        acc_signal, signal_colname_list, target_rate, window_size, window_overlap
    )
    time_2 = datetime.datetime.now()
    acc_epochs_info = {
        'epoch_indices': epoch_indices, 'epoch_stats': epoch_stats,
        'epoch_duration': window_size, 'epoch_overlap': window_overlap,
        'num_epochs': len(epoch_indices), 'signal_colnames': updated_signal_colname_list,
        'time_elapsed': (time_2 - time_1).total_seconds()
    }
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(
                f'{len(epoch_indices)} ACC epochs extracted '
                f'with duration: {window_size} seconds, overlap: {window_overlap} seconds.\n'
                f'Epoch analysis time elapsed: {(time_2 - time_1).total_seconds()} seconds.\n\n'
            )
    if output_file is not None:
        with open(output_file, 'wb') as f:
            pickle.dump({
                'acc_signal': acc_signal, 
                'acc_clean_info': acc_clean_info, 
                'acc_epochs_info': acc_epochs_info
            }, f)
    return acc_signal, acc_clean_info, acc_epochs_info


#####################################
### Integrate Multi-modal Signals ###
#####################################
def align_acc_to_ecg(ecg_signal: pd.Series, ecg_clean_info: dict, ecg_peaks_info: dict, 
                     acc_signal: pd.DataFrame, acc_clean_info: dict, window_size: int = 4,
                     log_file: str = None, output_file: str = None) -> pd.DataFrame | dict:
    ''' Align ACC signal to ECG by extracting ACC epochs centered around the ECG QRS complex. 
        ACC epoch size: window_size seconds.
        Return 
            a pd.DataFrame of aligned signal (columns stacked),
            a dictionary of ECG-ACC aligned epochs { ECG_QRS_Labels, ECG_QRS_Epochs },
                ECG_QRS_Epochs: { epoch_label: { 
                    'Index', 'Length', 'ECG_R_Peaks', ..., 'Vertical_Mean', ...
                } }.
    '''
    time_0 = datetime.datetime.now()
    # Read ECG, ACC info
    ecg_start_time = ecg_clean_info['start_time']
    ecg_sampling_rate = ecg_clean_info['sampling_rate']
    acc_start_time = acc_clean_info['start_time']
    acc_sampling_rate = acc_clean_info['resampled_rate']
    if ecg_sampling_rate != acc_sampling_rate:
        raise ValueError('ECG and ACC signals have different sampling rates!')
    # Compute the maximum shared time window between ECG and ACC signals
    # and extract signal subsets for alignment
    if ecg_start_time != acc_start_time:
        ecg_start_timestamp = datetime.datetime.strptime(ecg_start_time, '%Y%m%d_%H%M%S_%f')
        acc_start_timestamp = datetime.datetime.strptime(acc_start_time, '%Y%m%d_%H%M%S_%f')
        start_timestamp = max(ecg_start_timestamp, acc_start_timestamp)
        end_timestamp = min(ecg_signal.index[-1], acc_signal.index[-1])
        shared_duration = (end_timestamp - start_timestamp).total_seconds()
        if shared_duration < window_size:
            raise ValueError('The shared duration is shorter than the window size!')
        ## Extract ECG subset
        ecg_start_sample = (start_timestamp - ecg_start_timestamp).total_seconds() * ecg_sampling_rate
        ecg_end_sample = (end_timestamp - ecg_start_timestamp).total_seconds() * ecg_sampling_rate
        ecg_subset = ecg_signal.iloc[ecg_start_sample:ecg_end_sample]
        ## Extract ACC subset
        acc_start_sample = (start_timestamp - acc_start_timestamp).total_seconds() * acc_sampling_rate
        acc_end_sample = (end_timestamp - acc_start_timestamp).total_seconds() * acc_sampling_rate
        acc_subset = acc_signal.iloc[acc_start_sample:acc_end_sample]
    else:
        start_timestamp = datetime.datetime.strptime(ecg_start_time, '%Y%m%d_%H%M%S_%f')
        ecg_start_sample = 0
        ecg_end_sample = ecg_signal.shape[0] - 1
        acc_start_sample = 0
        acc_end_sample = acc_signal.shape[0] - 1
        ecg_subset = ecg_signal.copy()
        acc_subset = acc_signal.copy()
    # Double check subset size
    if ecg_subset.shape[0] != acc_subset.shape[0]:
        raise ValueError('ECG and ACC subsets have different lengths!')
    aligned_signal = pd.concat([ecg_subset, acc_subset], axis=1)
    aligned_dur = aligned_signal.shape[0] / (ecg_sampling_rate * 60)
    aligned_signal_info = {
        'start_time': start_timestamp, 'sampling_rate': ecg_sampling_rate, 
        'duration': aligned_dur, 'signal_columns': aligned_signal.columns.to_list()
    }
    # Iterate over ECG QRS epochs within the shared time window and extract ACC epochs
    # ecg_peaks_info = { 
    # 'r_peaks', 'r_peaks_indices', 'signal_quality',
    # 'all_epochs_info', 'good_epochs_info', ...
    # }
    ecg_qrs_epochs = ecg_peaks_info['good_epochs_info']
    acc_column_list = acc_clean_info['signal_colnames']
    aligned_epochs_info = {
        'ECG_QRS_Labels': np.zeros(aligned_signal.shape[0], dtype=np.int64),
        'ECG_QRS_Epochs': {},
    }
    epoch_label = 0
    for epoch_i, epoch_info in ecg_qrs_epochs['ECG_QRS_Epochs'].items():
        epoch_indices = epoch_info['Index']
        ## Check if the epoch is within the shared time window
        strict_ecg_start_sample = ecg_start_sample + window_size * ecg_sampling_rate // 2
        strict_ecg_end_sample = ecg_end_sample - window_size * ecg_sampling_rate // 2
        if (epoch_indices[0] >= strict_ecg_start_sample and epoch_indices[-1] <= strict_ecg_end_sample):
            epoch_label += 1
            corrected_epoch_indices = epoch_indices - ecg_start_sample
            aligned_epochs_info['ECG_QRS_Epochs'][epoch_label] = {
                'Index': corrected_epoch_indices,
            }
            ## Update label
            for i in corrected_epoch_indices:
                aligned_epochs_info['ECG_QRS_Labels'][i] = epoch_label
            ## Find heart rate
            epoch_rpeaks = epoch_info['ECG_R_Peaks']
            all_heart_rates = ecg_peaks_info['heart_rate']  # same index as rpeaks
            all_rpeaks = ecg_peaks_info['r_peaks_indices']
            ## If multiple R peaks in the epoch, use the median one
            if len(epoch_rpeaks) > 0:
                epoch_rpeak = epoch_rpeaks[len(epoch_rpeaks) // 2]
                epoch_heart_rate = all_heart_rates[np.where(all_rpeaks == epoch_rpeak)[0][0]]
            else:
                epoch_rpeak = np.nan
                epoch_heart_rate = np.nan
            aligned_epochs_info['ECG_QRS_Epochs'][epoch_label]['Heart_Rate'] = epoch_heart_rate
            aligned_epochs_info['ECG_QRS_Epochs'][epoch_label]['Center_R_Peak'] = epoch_rpeak
            ## Find all waves
            for wave_name, wave_indices in epoch_info.items():
                if 'ECG' in wave_name:
                    aligned_epochs_info['ECG_QRS_Epochs'][epoch_label][wave_name] = wave_indices - ecg_start_sample
            ## Compute ACC epoch stats
            # acc_epoch_center_index = corrected_epoch_indices[len(epoch_indices) // 2]
            acc_epoch_center_index = epoch_rpeak
            acc_epoch_start_index = acc_epoch_center_index - ecg_sampling_rate * window_size // 2
            acc_epoch_end_index = acc_epoch_center_index + ecg_sampling_rate * window_size // 2
            if acc_epoch_start_index < 0 or acc_epoch_end_index >= acc_signal.shape[0]:
                print(f'For ECG Epoch {epoch_i}, ACC epoch is out of range: '
                      f'[{acc_epoch_start_index}, {acc_epoch_end_index}]!')
                continue
            for acc_col in acc_column_list:
                acc_col_epoch = acc_subset[acc_col][acc_epoch_start_index:acc_epoch_end_index]
                aligned_epochs_info['ECG_QRS_Epochs'][epoch_label][f'{acc_col}_Mean'] = acc_col_epoch.mean()
                aligned_epochs_info['ECG_QRS_Epochs'][epoch_label][f'{acc_col}_Variance'] = acc_col_epoch.std()
    num_aligned_epochs = len(aligned_epochs_info['ECG_QRS_Epochs'])
    print(f'{datetime.datetime.now()}: {num_aligned_epochs} epochs extracted.')
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(f'{aligned_signal.shape[0]} samples aligned.\n')
            f.write(f'Sampling rate: {ecg_sampling_rate} Hz, ')
            f.write(f'duration: {aligned_dur} minutes.\n')
            f.write(f'Signal columns: {aligned_signal.columns.to_list()}\n')
            f.write(f'{num_aligned_epochs} epochs extracted.\n')
            f.write(f'Epoch info dict keys: {list(aligned_epochs_info["ECG_QRS_Epochs"][1].keys())}\n')
            f.write(f'Time elapsed: {(datetime.datetime.now() - time_0).total_seconds()} seconds.\n\n')
    # Save aligned signal and epochs info
    if output_file is not None:
        with open(output_file, 'wb') as f:
            pickle.dump({
                'aligned_signal': acc_signal, 
                'aligned_signal_info': aligned_signal_info,
                'aligned_epochs_info': aligned_epochs_info, 
            }, f)
    return aligned_signal, aligned_signal_info, aligned_epochs_info

