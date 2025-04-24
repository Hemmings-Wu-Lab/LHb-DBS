import h5py
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
import matplotlib.pyplot as plt
from ace_tools import display_dataframe_to_user

# File paths and sampling rate
record_file = '/mnt/data/trial information IAPS_L_-0.999.mat'
trial_file = '/mnt/data/IAPS_L_-0.999.mat'
fs = 23241.0

# Design bandpass filter (300–3000 Hz)
low, high = 300.0, 3000.0
b, a = butter(3, [low/(fs/2), high/(fs/2)], btype='band')

# Load data, detect spikes, compute durations and rates
data = []
with h5py.File(record_file, 'r') as f_rec, h5py.File(trial_file, 'r') as f_tr:
    ans_refs = f_rec['record']['patient_ans'][:]
    trial_refs = f_rec['record']['trial_number'][:]
    spk_refs = f_tr['Trial']['SPK_01'][:]
    
    for i in range(len(spk_refs)):
        # Extract raw signal
        raw_ref = spk_refs[i][0]
        raw = f_tr[raw_ref][:].flatten()
        
        # Compute duration from sample count
        duration = raw.size / fs
        
        # Bandpass filter
        sig = filtfilt(b, a, raw)
        
        # Spike detection: threshold = 4×SD, refractory period = 1 ms
        thr = np.std(sig) * 4
        peaks, _ = find_peaks(sig, height=thr, distance=int(fs * 0.001))
        spike_count = peaks.size
        
        # Retrieve metadata
        ans_ref = ans_refs[i][0]
        patient_ans = f_rec[ans_ref][()].item() if isinstance(ans_ref, h5py.Reference) else ans_ref
        tn_ref = trial_refs[i][0]
        trial_num = f_rec[tn_ref][()].item() if isinstance(tn_ref, h5py.Reference) else tn_ref
        
        # Compute spike rate (Hz)
        spike_rate = spike_count / duration
        
        data.append({
            'trial_number': int(trial_num),
            'patient_ans': patient_ans,
            'duration_s': duration,
            'spike_count': spike_count,
            'spike_rate_hz': spike_rate
        })

# Build DataFrame
df_rates = pd.DataFrame(data)
display_dataframe_to_user('spike_rate_per_trial', df_rates)

# Compute mean and SEM by patient_ans
summary_sem = df_rates.groupby('patient_ans')['spike_rate_hz'].agg(
    avg_spike_rate_hz='mean',
    sem_spike_rate_hz=lambda x: x.sem()
).reset_index()
display_dataframe_to_user('sem_spike_rate_by_patient_ans', summary_sem)

# Plot bar chart with SEM error bars
fig, ax = plt.subplots()
ax.bar(summary_sem['patient_ans'].astype(str), summary_sem['avg_spike_rate_hz'])
ax.errorbar(summary_sem['patient_ans'].astype(str), 
            summary_sem['avg_spike_rate_hz'], 
            yerr=summary_sem['sem_spike_rate_hz'], 
            fmt='none', capsize=5)
ax.set_xlabel('patient_ans')
ax.set_ylabel('Average spike rate (Hz)')
ax.set_title('Average Spike Rate ± SEM by Patient Answer')
plt.show()
