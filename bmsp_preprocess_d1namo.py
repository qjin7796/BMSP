# Preprocess ECG signal raw data (csv) from D1NAMO dataset:
#   signal cleaning: fill missing values, correct outliers, denoising.
#   signal quality: locate R peaks, compute signal quality index.
#   heart rate: compute heart rate and heart rate variability.
#   extract epochs: segment signal around R peaks, delineate all QRS waves.
# ----------------------------------------------------------------------------------------
# Preprocess accelerometer signal raw data (csv) from D1NAMO dataset:
#   signal cleaning: (calibration,) fill missing values, correct outliers, denoising.
#       NB: D1NAMO dataset does not provide the ground truth for calibration. 
#   signal resampling: resample signal to match the ECG singal sampling rate.
#   vector magnitude: compute the vector magnitude of the 3-axis signal and its variance.
#   extract epochs: segment signal into short overlapping epochs, compute epoch stats.
# ----------------------------------------------------------------------------------------
# ECG - acceleration alignment:
#   for each ECG epoch, extract an acceleration epoch centered at the R peak.
#       NB: typical R-R interval is 0.6-1.2 seconds.
# Author: Qiuhan Jin
# Date: 2024-08-20


from bmsp_preprocess_utils import *


if __name__ == '__main__':
    # Example subject data
    subject_type = 'healthy'
    # subject_type = 'diabetes'
    subject = '001'
    ## ECG
    ecg_data_type = 'ecg'
    ecg_signal_name = 'ECG'
    ecg_signal_label = 'ecg'
    ecg_signal_colname = 'EcgWaveform'
    ecg_timestamp_colname = 'Time'
    subject_ecg_sample_time = '2014_10_01-12_50_01'
    # subject_ecg_sample_time = '2014_10_04-06_34_57'
    ## ACC
    acc_data_type = 'sensor'
    acc_signal_name = 'Accel'
    acc_signal_label = 'acc'
    acc_signal_colname_list = ['Vertical', 'Lateral', 'Sagittal']
    acc_timestamp_colname = 'Time'
    subject_acc_sample_time = '2014_10_01-12_50_01'
    # subject_acc_sample_time = '2014_10_04-06_34_57'

    # Specify data path
    data_dir = '/Users/qjin/Downloads/D1NAMO_subset'
    output_dir = os.path.join(data_dir, 'preprocessed_data')
    ## ECG
    ecg_signal_file_path = os.path.join(
        data_dir, f'{subject_type}_subset_{ecg_data_type}_data', 
        subject, 'sensor_data', subject_ecg_sample_time, f'{subject_ecg_sample_time}_{ecg_signal_name}.csv'
    )
    ecg_output_dir = os.path.join(output_dir, 'ecg')
    if not os.path.exists(ecg_output_dir):
        os.makedirs(ecg_output_dir)
    ecg_preproc_output_dir = os.path.join(
        ecg_output_dir, f'{subject_type}_subset_{ecg_signal_label}_data', 
        subject, subject_ecg_sample_time
    )
    if not os.path.exists(ecg_preproc_output_dir):
        os.makedirs(ecg_preproc_output_dir)
    ecg_preproc_output_file_path = os.path.join(
        ecg_preproc_output_dir, 
        f'{subject_ecg_sample_time}_{ecg_signal_label}_preprocessing.pkl'
    )
    ## ACC
    acc_signal_file_path = os.path.join(
        data_dir, f'{subject_type}_subset_{acc_data_type}_data', 
        subject, 'sensor_data', subject_acc_sample_time, f'{subject_acc_sample_time}_{acc_signal_name}.csv'
    )
    acc_output_dir = os.path.join(output_dir, 'accelerometer')
    if not os.path.exists(acc_output_dir):
        os.makedirs(acc_output_dir)
    acc_preproc_output_dir = os.path.join(
        acc_output_dir, f'{subject_type}_subset_{acc_signal_label}_data', 
        subject, subject_acc_sample_time
    )
    if not os.path.exists(acc_preproc_output_dir):
        os.makedirs(acc_preproc_output_dir)
    acc_preproc_output_file_path = os.path.join(
        acc_preproc_output_dir, 
        f'{subject_acc_sample_time}_{acc_signal_label}_preprocessing.pkl'
    )
    ## ECG - ACC alignment
    ecg_acc_output_dir = os.path.join(output_dir, 'ecg_acc_alignment')
    if not os.path.exists(ecg_acc_output_dir):
        os.makedirs(ecg_acc_output_dir)
    ecg_acc_preproc_output_dir = os.path.join(
        ecg_acc_output_dir, f'{subject_type}_subset_ecg_acc_data', 
        subject, subject_ecg_sample_time
    )
    if not os.path.exists(ecg_acc_preproc_output_dir):
        os.makedirs(ecg_acc_preproc_output_dir)
    ecg_acc_preproc_output_file_path = os.path.join(
        ecg_acc_preproc_output_dir, 
        f'{subject_ecg_sample_time}_ecg_acc_alignment.pkl'
    )

    # # Start preprocessing
    # log_time = datetime.datetime.now()
    # log_time_str = log_time.strftime('%Y%m%d_%H%M%S')
    # preproc_log = os.path.join(output_dir, f'd1namo_preprocessing_log_{log_time_str}.txt')
    # with open(preproc_log, 'w') as f:
    #     f.write(f'Preprocessing log: {datetime.datetime.now()}\n\n')
    # ## ECG
    # if not os.path.exists(ecg_signal_file_path):
    #     raise FileNotFoundError(f'{ecg_signal_file_path} does not exist!')
    # else:
    #     print(f'{datetime.datetime.now()}: '
    #           f'Preprocessing ECG signal file: {ecg_signal_file_path}')
    #     with open(preproc_log, 'a') as f:
    #         f.write(f'\nPreprocessing ECG signal file: {ecg_signal_file_path}\n')
    #     # Read and preprocess signal from the csv file
    #     ecg_cleaned, ecg_clean_info, ecg_peaks_info = preproc_ecg(
    #         file_path = ecg_signal_file_path, 
    #         timestamp_colname = ecg_timestamp_colname,
    #         ecg_colname = ecg_signal_colname,
    #         outlier_exclude = 2, outlier_side = 'both',
    #         denoise_method = 'neurokit', 
    #         quality_threshold = 0.8,
    #         group_epochs_window = 180,
    #         log_file = preproc_log,
    #         output_file = ecg_preproc_output_file_path
    #     )
    #     print(f'{datetime.datetime.now()}: ECG signal preprocessed '
    #           f'and saved to {ecg_preproc_output_file_path}!')
    # ## ACC
    # if not os.path.exists(acc_signal_file_path):
    #     raise FileNotFoundError(f'{acc_signal_file_path} does not exist!')
    # else:
    #     print(f'{datetime.datetime.now()}: '
    #           f'Preprocessing accelerometer signal file: {acc_signal_file_path}')
    #     with open(preproc_log, 'a') as f:
    #         f.write(f'\nPreprocessing accelerometer signal file: {acc_signal_file_path}\n')
    #     # Read and preprocess signal from the csv file
    #     acc_cleaned, acc_clean_info, acc_epochs_info = preproc_acc(
    #         file_path = acc_signal_file_path, 
    #         timestamp_colname = acc_timestamp_colname,
    #         signal_colname_list = acc_signal_colname_list,
    #         target_rate = 250, lowcut = 0.5, highcut = 10.,
    #         window_size = 5, window_overlap = 4,
    #         log_file = preproc_log,
    #         output_file = acc_preproc_output_file_path
    #     )
    #     print(f'{datetime.datetime.now()}: ACC signal preprocessed '
    #           f'and saved to {acc_preproc_output_file_path}!')
    # ## ECG - ACC alignment
    # with open(preproc_log, 'a') as f:
    #     f.write(f'\nAligning ECG and accelerometer signals...\n')
    # ecg_acc_signal, ecg_acc_signal_info, ecg_acc_epochs_info = align_acc_to_ecg(
    #     ecg_cleaned, ecg_clean_info, ecg_peaks_info, 
    #     acc_cleaned, acc_clean_info, 
    #     window_size = 4, log_file = preproc_log, 
    #     output_file = ecg_acc_preproc_output_file_path
    # )
    # print(f'{datetime.datetime.now()}: ECG and accelerometer signals aligned '
    #       f'and saved to {ecg_acc_preproc_output_file_path}!')
    # print(f'{datetime.datetime.now()}: Preprocessing completed!')
    # print(f'Time elapsed: {datetime.datetime.now() - log_time}')

    with open(ecg_preproc_output_file_path, 'rb') as f:
        ecg_data = pickle.load(f)
    with open(acc_preproc_output_file_path, 'rb') as f:
        acc_data = pickle.load(f)
    print(f'{datetime.datetime.now()}: '
          f'Aligning ECG and accelerometer signals...')
    ecg_acc_signal, ecg_acc_signal_info, ecg_acc_epochs_info = align_acc_to_ecg(
        ecg_data['ecg_signal'], ecg_data['ecg_clean_info'], ecg_data['ecg_peaks_info'], 
        acc_data['acc_signal'], acc_data['acc_clean_info'], 
        window_size = 4, log_file = None, 
        output_file = ecg_acc_preproc_output_file_path
    )
