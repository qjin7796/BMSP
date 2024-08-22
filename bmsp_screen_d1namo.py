# For each type of signal, screen number of subjects/sessions and data size
# Author: Qiuhan Jin
# Date: 2024-08-16


import os, pandas as pd, datetime


# Dataset directory
data_dir = '/Users/qjin/Downloads/D1NAMO_subset'
# Subdirectories:
# - <subject type>_subset_<data type>_data
# -- 001
# --- sensor_data
# ---- <time>
# ----- <time>_<signal name>.csv
subject_dict = {
    'healthy': ['001'],
    'diabetes': ['001'],
}
# Name format:
# 1st key: data type (folder name), 
# 2nd key: signal name (csv file name), 
# value: customized string
data_type_dict = {
    'sensor': {
        'Accel': 'acc',
        'Breathing': 'bre',
    },
    'ecg': {
        'ECG': 'ecg',
    },
}


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    start_time_str = start_time.strftime('%Y%m%d_%H%M%S')
    screen_log = os.path.join(data_dir, f'screening_log_{start_time_str}.txt')
    # List data types, signal types, subject types, subjects
    with open(screen_log, 'w') as f:
        f.write(f'Data screening log: {start_time}\n')
        f.write(f'{len(data_type_dict)} data folders: {list(data_type_dict.keys())}\n')
        for data_type, signal_dict in data_type_dict.items():
            f.write(f'  {len(signal_dict)} {data_type} signals: {list(signal_dict.keys())}\n')
        f.write(f'{len(subject_dict)} subject types: {list(subject_dict.keys())}\n')
        for subject_type, subject_list in subject_dict.items():
            f.write(f'  {len(subject_list)} {subject_type} subjects: {subject_list}\n')
            # List number of data
            for data_type, signal_dict in data_type_dict.items():
                f.write(f'    {data_type} data:\n')
                for subject in subject_list:
                    subject_dir = os.path.join(
                        data_dir, f'{subject_type}_subset_{data_type}_data', 
                        subject, 'sensor_data'
                    )
                    if not os.path.exists(subject_dir):
                        f.write(f'      {subject}: 0\n')
                        continue
                    files = os.listdir(subject_dir)
                    files = [f for f in files if not f.startswith('.')]
                    f.write(f'      {subject}: {len(files)}\n')
        f.write('\n')
    # Screen signal csv files
    for data_type, signal_dict in data_type_dict.items():
        for subject_type, subject_list in subject_dict.items():
            for subject in subject_list:
                subject_dir = os.path.join(
                    data_dir, f'{subject_type}_subset_{data_type}_data', 
                    subject, 'sensor_data'
                )
                if not os.path.exists(subject_dir):
                    with open(screen_log, 'a') as f:
                        f.write(f'{subject_dir} does not exist!\n\n')
                    continue
                files = os.listdir(subject_dir)
                files = [f for f in files if not f.startswith('.')]
                for time_name in files:
                    for signal_name in signal_dict.keys():
                        signal_file_path = os.path.join(
                            subject_dir, time_name, f'{time_name}_{signal_name}.csv'
                        )
                        if not os.path.exists(signal_file_path):
                            with open(screen_log, 'a') as f:
                                f.write(f'{signal_file_path} does not exist!\n\n')
                            continue
                        # Read the csv file
                        df = pd.read_csv(signal_file_path, sep=',', header=0)
                        # Print size
                        with open(screen_log, 'a') as f:
                            # f.write(f'Size: {df.shape}\n  Dtypes:\n{df.dtypes}\n\n')
                            f.write(f'{signal_file_path}: {df.shape}\n')
                            f.write(f'{df.dtypes}\n\n')
                    break
    with open(screen_log, 'a') as f:
        # Print time elapsed in seconds
        f.write(f'\n\nTime elapsed: {(datetime.datetime.now() - start_time).seconds} seconds\n\n')
