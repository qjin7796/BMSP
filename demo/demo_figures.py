from bmsp_preprocess_utils import *
from neurokit2.stats import rescale
from matplotlib.gridspec import GridSpec


# Read data
hecg_path = '/Users/qjin/Downloads/D1NAMO_subset/preprocessed_data/ecg/healthy_subset_ecg_data/001/2014_10_01-12_50_01/2014_10_01-12_50_01_ecg_preprocessing.pkl'
hecg_raw_path = '/Users/qjin/Downloads/D1NAMO_subset/healthy_subset_ecg_data/001/sensor_data/2014_10_01-12_50_01/2014_10_01-12_50_01_ECG.csv'
hacc_path = '/Users/qjin/Downloads/D1NAMO_subset/preprocessed_data/accelerometer/healthy_subset_acc_data/001/2014_10_01-12_50_01/2014_10_01-12_50_01_acc_preprocessing.pkl'
hacc_raw_path = '/Users/qjin/Downloads/D1NAMO_subset/healthy_subset_sensor_data/001/sensor_data/2014_10_01-12_50_01/2014_10_01-12_50_01_Accel.csv'
with open(hecg_path, 'rb') as f:
    hecg = pickle.load(f)
with open(hacc_path, 'rb') as f:
    hacc = pickle.load(f)
hecg_raw = pd.read_csv(hecg_raw_path, header=0, sep=',')
hacc_raw = pd.read_csv(hacc_raw_path, header=0, sep=',')
aligned_ecg_acc_file = '/Users/qjin/Downloads/D1NAMO_subset/preprocessed_data/ecg_acc_alignment/healthy_subset_ecg_acc_data/001/2014_10_01-12_50_01/2014_10_01-12_50_01_ecg_acc_alignment.pkl'
with open(aligned_ecg_acc_file, 'rb') as f:
    aligned_ecg_acc = pickle.load(f)
output_dir = '/Users/qjin/Documents/GitHub/BMSP/demo/figures'


# Plot ECG signal cleaning
# first 3 minutes (3*60*250 samples), 3 minutes in the middle, and last 3 minutes
segment_length = 3*60*250
num_samples = hecg_raw.shape[0]
segments = [
    (0, segment_length),
    (num_samples//2-segment_length//2, num_samples//2+segment_length//2),
    (num_samples-segment_length, num_samples)
]
fig, axes = plt.subplots(3, 1, figsize=(20, 8))
for i, segment in enumerate(segments):
    # Read data
    data_index = [x for x in range(segment[0], segment[1])]
    raw_signal = hecg_raw['EcgWaveform'][data_index].to_numpy() - \
        hecg['ecg_clean_info']['raw_stats']['mean']
    cleaned_signal = hecg['ecg_signal'][data_index].to_numpy()
    signal_quality = hecg['ecg_peaks_info']['signal_quality'][data_index].to_numpy()
    ax = axes[i]
    # Plot signal quality area first
    ax.set_ylim(-2000, 2000)
    quality_min = -2000
    quality_max = 2000
    quality = rescale(signal_quality, to=[quality_min, quality_max])
    ax.fill_between(
        data_index, quality_min, quality, alpha=0.12, zorder=0,
        interpolate=True, facecolor="#4CAF50", label="Signal quality",
    )
    ax.hlines(2000*0.8, segment[0], segment[1], linestyles='dashed', 
              color="#4CAF50", label="Signal quality threshold", zorder=1)
    # Plot raw signal
    ax.plot(data_index, raw_signal, color="#B0BEC5", label="Raw signal", zorder=1)
    # Plot cleaned signal
    ax.plot(data_index, cleaned_signal, color="#F44336", label="Cleaned signal", 
            zorder=3, linewidth=1)
    # Change x_ticks to once per 20 seconds
    x_ticks = [x for x in range(segment[0], segment[1] + 250*20, 250*20)]
    x_labels = [x//250 for x in x_ticks]
    ax.set_xticks(x_ticks, x_labels)
    # Add title, labels, and legend
    if i == 0:
        ax.legend(loc='upper right')
        ax.set_title('ECG Signal Cleaning', fontweight='bold', fontsize=20)
    if i == 1:
        ax.set_ylabel('Amplitude', fontsize=16)
    if i == 2:
        ax.set_xlabel('Time (seconds)', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'ecg_cleaning.png'))
plt.close()


# Plot ECG signal QRS epochs
# a 1-min window with signal of good quality 15900-15960 second
num_samples = hecg_raw.shape[0]
segment = (15800*250, 15860*250)
fig = plt.figure(constrained_layout=False, figsize=(20, 8))
fig.suptitle('ECG Epochs', fontweight='bold', fontsize=20)
gs = GridSpec(2, 2, width_ratios=[2 / 3, 1 / 3])
ax0 = fig.add_subplot(gs[0, :-1])  # signal
ax1 = fig.add_subplot(gs[1, :-1], sharex=ax0)  # heart rate
ax2 = fig.add_subplot(gs[:, -1])  # epochs
# Read data
data_index = [x for x in range(segment[0], segment[1])]
cleaned_signal = hecg['ecg_signal'][data_index].to_numpy()
cleaned_signal_df = hecg['ecg_signal'].iloc[data_index]
all_rpeaks = hecg['ecg_peaks_info']['r_peaks']['ECG_R_Peaks'].to_numpy()
rpeaks_indices = hecg['ecg_peaks_info']['r_peaks_indices'].tolist()
heart_rate = hecg['ecg_peaks_info']['heart_rate']
selected_qrs_epochs = hecg['ecg_peaks_info']['good_epochs_info']['ECG_QRS_Labels']
selected_rpeaks = []
selected_qrs_labels = []
epoch_rpeaks = {}
for x in range(segment[0], segment[1]):
    if all_rpeaks[x] > 0 and selected_qrs_epochs[x] > 0:
        selected_rpeaks.append(x)
        selected_qrs_labels.append(selected_qrs_epochs[x])
        epoch_rpeaks[selected_qrs_epochs[x]] = x
selected_rpeaks_indices = [rpeaks_indices.index(x) for x in selected_rpeaks]
selected_heart_rate = heart_rate[selected_rpeaks_indices]
# Plot cleaned signal with R-peaks
ax0.plot(data_index, cleaned_signal, color="#F44336", label="Cleaned signal", 
         zorder=3, linewidth=1)
# Plot R-peaks
ax0.scatter(selected_rpeaks, hecg['ecg_signal'][selected_rpeaks].to_numpy(), 
            color="#FFC107", label="R-peaks", zorder=2)
# Add title, labels, and legend
ax0.set_ylim(-100, 200)
ax0.legend(loc='upper right')
ax0.set_ylabel('Amplitude', fontsize=10)
ax0.set_title('ECG Signal', fontsize=12)
# Plot heart rate with scatters marked at R-peaks
ax1.plot(selected_rpeaks, selected_heart_rate, color="#F44336", label="Heart rate", zorder=2)
ax1.scatter(selected_rpeaks, selected_heart_rate, color="#FFC107", label="Heart rate", zorder=3)
# Change x_ticks to once per 10 seconds
x_ticks = [x for x in range(segment[0], segment[1] + 250*10, 250*10)]
x_labels = [x//250 for x in x_ticks]
ax1.set_xticks(x_ticks, x_labels)
ax1.set_xlabel('Time (seconds)', fontsize=10)
ax1.set_ylabel('BPM\n', fontsize=10)
ax1.set_title('Heart Rate', fontsize=12)
# Plot epochs
_ = nk.ecg_segment(cleaned_signal_df, sampling_rate=250, show=True, ax=ax2)
# Plot P, Q, S, T waves
all_epoch_labels = list(hecg['ecg_peaks_info']['good_epochs_info']['ECG_QRS_Epochs'].keys())
wave_key = ['ECG_P_Peaks', 'ECG_Q_Peaks', 'ECG_S_Peaks', 'ECG_T_Peaks']
wave_color = ["#3949AB", "#1E88E5", "#039BE5",  "#00ACC1"]
wave_name = ['P-waves', 'Q-waves', 'S-waves', 'T-waves']
scatter_labeled = [False, False, False, False]
for epoch_i in selected_qrs_labels:
    indices = hecg['ecg_peaks_info']['good_epochs_info']['ECG_QRS_Epochs'][epoch_i]['Index']
    rpeak_index = epoch_rpeaks[epoch_i]
    for i, key in enumerate(wave_key):
        peak_indices = hecg['ecg_peaks_info']['good_epochs_info']['ECG_QRS_Epochs'][epoch_i][key]
        if len(peak_indices) == 0:
            continue
        x_axis = (peak_indices-rpeak_index) / 250
        y_axis = hecg['ecg_signal'][peak_indices].to_numpy()
        if not all(scatter_labeled):
            ax2.scatter(x_axis, y_axis, color=wave_color[i], label=wave_name[i], 
                        marker='+', alpha=1, zorder=3)
            scatter_labeled[i] = True
        else:
            ax2.scatter(x_axis, y_axis, color=wave_color[i], marker='+', alpha=0.5, zorder=3)
ax2.legend(loc='upper right')
plt.savefig(os.path.join(output_dir, 'ecg_epochs.png'))
plt.close()


# Plot ACC signal cleaning
# first 2 minutes (1*60*100/250 samples), 2 minutes in the middle, and last 2 minutes
# NB: raw signal sampling rate = 100 Hz, cleaned signal sampling rate = 250 Hz
columns = ['Vertical', 'Lateral', 'Sagittal']
colors = ['red', 'green', 'royalblue']
fig, axes = plt.subplots(3, 3, figsize=(20, 8))

segment_length = 2*60*1000
raw_factor = 10
cleaned_factor = 4
num_samples = hacc['acc_signal'].shape[0]
segments = [
    (0, segment_length),
    (num_samples//2-segment_length//2, num_samples//2+segment_length//2),
    (num_samples-segment_length, num_samples)
]
for i, segment in enumerate(segments):
    # Raw data
    raw_data_index = [x for x in range(int(segment[0]/raw_factor), int(segment[1]/raw_factor))]
    raw_signal = hacc_raw[columns].iloc[raw_data_index]
    # Cleaned data
    cleaned_data_index = [x for x in range(int(segment[0]/cleaned_factor), int(segment[1]/cleaned_factor))]
    cleaned_signal = hacc['acc_signal'][columns].iloc[cleaned_data_index]

    for j, col in enumerate(columns):
        raw_x = [x for x in range(segment[0], segment[1], raw_factor)]
        # raw_y = raw_signal[col].to_numpy() - hacc['acc_clean_info']['raw_stats']['mean'][j]
        raw_y = raw_signal[col].to_numpy() - raw_signal[col].mean()
        cleaned_x = [x for x in range(segment[0], segment[1], cleaned_factor)]
        cleaned_y = cleaned_signal[col].to_numpy()
        ax = axes[i, j]
        ax.plot(raw_x, raw_y, color='black', label=f'Raw-{col}', zorder=1, alpha=0.3)
        ax.plot(cleaned_x, cleaned_y, color=colors[j], label=f'Cleaned-{col}', zorder=2, alpha=0.75)
        # Change x_ticks to once per 10 seconds
        x_ticks = [x for x in range(segment[0], segment[1] + 1000*10, 1000*10)]
        x_labels = [x//1000 for x in x_ticks]
        ax.set_xticks(x_ticks, x_labels)
        # Add title, labels, and legend
        if i == 0:
            ax.legend(loc='upper right')
            if j == 1:
                ax.set_title('ACC Signal Cleaning', fontweight='bold', fontsize=20)
        if i == 1 and j == 0:
            ax.set_ylabel('Amplitude', fontsize=16)
        if i == 2 and j == 1:
            ax.set_xlabel('Time (seconds)', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'acc_cleaning.png'))
plt.close()


# Plot ACC vector features
# first 2 minutes (1*60*100/250 samples), 2 minutes in the middle, and last 2 minutes
# NB: raw signal sampling rate = 100 Hz, cleaned signal sampling rate = 250 Hz
columns = ['Vector_Magnitude', 'Vector_Magnitude_Gradient']
colors = ['orangered', 'orange']
fig, axes = plt.subplots(3, 2, figsize=(16, 6))

segment_length = 2*60*250
num_samples = hacc['acc_signal'].shape[0]
segments = [
    (0, segment_length),
    (num_samples//2-segment_length//2, num_samples//2+segment_length//2),
    (num_samples-segment_length, num_samples)
]
for i, segment in enumerate(segments):
    # Cleaned data
    cleaned_data_index = [x for x in range(int(segment[0]), int(segment[1]))]
    cleaned_signal = hacc['acc_signal'][columns].iloc[cleaned_data_index]
    for j, col in enumerate(columns):
        cleaned_x = [x for x in range(segment[0], segment[1])]
        cleaned_y = cleaned_signal[col].to_numpy()
        ax = axes[i,j]
        ax.plot(cleaned_x, cleaned_y, color=colors[j], label=col)
        # Change x_ticks to once per 10 seconds
        x_ticks = [x for x in range(segment[0], segment[1] + 250*10, 250*10)]
        x_labels = [x//250 for x in x_ticks]
        ax.set_xticks(x_ticks, x_labels)
        # Add title, labels, and legend
        if i == 0:
            ax.legend(loc='upper right')
        if i == 1:
            ax.set_ylabel('Amplitude', fontsize=16)
        if i == 2:
            ax.set_xlabel('Time (seconds)', fontsize=16)
plt.suptitle('ACC Vector Features', fontweight='bold', fontsize=20)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'acc_vector_magnitude.png'))
plt.close()


# Plot ACC-ECG aligned epochs
# Plot ECG epochs (signal, R peaks, heart rate, 4 waves)
# Plot ACC vector magnitude variance
segment = (3917600, 3920000)  # 10 seconds
x_axis = np.arange(segment[0], segment[1], 1)
ecg_signal = hecg['ecg_signal'][segment[0]:segment[1]].to_numpy()  # 10 secs
wave_key = ['ECG_P_Peaks', 'ECG_Q_Peaks', 'ECG_S_Peaks', 'ECG_T_Peaks']
wave_color = ["#3949AB", "#1E88E5", "#039BE5",  "#00ACC1"]
wave_name = ['P-waves', 'Q-waves', 'S-waves', 'T-waves']
segment_info = {
    'Center_R_Peaks': [],
    'Heart_Rate': [],
    'ACC_VMV': [],
}
for wave_key_i in wave_key:
    segment_info[wave_key_i] = []
all_epochs = aligned_ecg_acc['aligned_epochs_info']['ECG_QRS_Epochs']  # 30k+ dict
for epoch_i, epoch_info in all_epochs.items():
    epoch_indices = epoch_info['Index']
    epoch_center_rpeak = epoch_info['Center_R_Peak']
    if epoch_center_rpeak > segment[0] and epoch_center_rpeak < segment[1]:
        segment_info['Center_R_Peaks'].append(epoch_center_rpeak)
        segment_info['Heart_Rate'].append(epoch_info['Heart_Rate'])
        segment_info['ACC_VMV'].append(epoch_info['Vector_Magnitude_Variance'])
        for i, wave in enumerate(wave_key):
            wave_indices = epoch_info[wave]
            wave_indices_available = np.intersect1d(wave_indices, x_axis)
            segment_info[wave].append(wave_indices_available)
rpeaks_signal = hecg['ecg_signal'][segment_info['Center_R_Peaks']].to_numpy()

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})
ax1 = axes[0]
ax2 = axes[1]
ax3 = axes[2]
# Plot cleaned signal with R-peaks
ax1.plot(x_axis, ecg_signal, color="#F44336", label="ECG signal", zorder=1, linewidth=1)
# Plot R-peaks
ax1.scatter(segment_info['Center_R_Peaks'], rpeaks_signal, 
            color="#FFC107", label="R-peaks", zorder=4)
# Plot heart rate with scatters marked at R-peaks
ax2.plot(segment_info['Center_R_Peaks'], segment_info['Heart_Rate'], color="#F44336", label="Heart rate")
ax2.scatter(segment_info['Center_R_Peaks'], segment_info['Heart_Rate'], color="#FFC107")
# Plot P, Q, S, T waves
for w, wave_name in enumerate(wave_key):
    wave_indices = []
    for item in segment_info[wave_name]:
        wave_indices.extend(item.tolist())
    ax1.scatter(wave_indices, hecg['ecg_signal'][wave_indices], 
                color=wave_color[w], label=wave_name, zorder=3)
# Plot ACC-epoch vector magnitude variance
ax3.plot(segment_info['Center_R_Peaks'], segment_info['ACC_VMV'], color="#4CAF50", 
         label="ACC-epoch vector magnitude variance")
ax3.scatter(segment_info['Center_R_Peaks'], segment_info['ACC_VMV'], color="#FFC107")
# Change x_ticks to once per 1 second
x_ticks = [x for x in range(segment[0], segment[1]+250, 250)]
x_labels = [x//250 for x in x_ticks]
ax1.set_ylim(-425, 400)
ax1.legend(loc='lower right')
ax2.legend(loc='lower right')
ax3.legend(loc='lower right')
ax3.set_xticks(x_ticks, x_labels)
ax3.set_xlim(x_ticks[0]-75, x_ticks[-1]-75)
ax3.set_xlabel('Time (seconds)', fontsize=12)
ax3.set_ylabel('Vector magnitude variance', fontsize=12)
ax2.set_ylabel('Heart rate (bpm)', fontsize=12)
ax1.set_ylabel('Signal magnitude', fontsize=12)
plt.suptitle('ECG-ACC Aligned Epochs', fontsize=20, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'ECG_ACC_aligned_epochs.png'))
plt.close()