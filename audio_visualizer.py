import os
import numpy as np
import soundfile as sf
from scipy import signal
from matplotlib import pyplot as plt

def pow_two_pad_and_window(vec, fs, show=False):
    window = signal.windows.tukey(len(vec), alpha=0.3)
    windowed_vec = vec * window
    padded_windowed_vec = np.pad(windowed_vec, (0, 2**int(np.ceil(np.log2(len(windowed_vec)))) - len(windowed_vec)))
    if show:
        dur = len(padded_windowed_vec) / fs
        t = np.linspace(0, dur, len(padded_windowed_vec))
        plt.figure()
        plt.plot(t, padded_windowed_vec)
        plt.show()
    return padded_windowed_vec/max(padded_windowed_vec)

os.chdir(os.path.dirname(os.path.abspath(__file__)))

plt.rcParams['text.usetex'] = True

x, fs = sf.read('./doa_data/audio/capon_20250429_18-06-59.wav')

print(fs)

dur = 3e-3
hi_freq = 60e3
low_freq = 20e3

t_tone = np.linspace(0, dur, int(fs*dur))
chirp = signal.chirp(t_tone, hi_freq, t_tone[-1], low_freq)    
sig = pow_two_pad_and_window(chirp, fs, show=False)

t = np.linspace(0, len(x)/fs, len(x))

x_mono = x.T[0]

xcorr = np.roll(signal.correlate(x_mono, sig, 'same'), -len(sig)//2)

env = np.abs(signal.hilbert(xcorr))

peaks_indexes = signal.find_peaks(env, prominence=10)[0]

plt.figure()
plt.plot(t, x_mono, color='black', linewidth=0.8)
plt.xlabel('time [s]', fontsize=16)
plt.ylabel('Amplitude', fontsize=16)
plt.ylim([-1.1*max(np.abs(x_mono)), 1.1*max(np.abs(x_mono))])
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('Recorded audio', fontsize=20)
plt.grid()
plt.show()

plt.figure()
plt.plot(t, xcorr, color='black', linewidth=0.8)
plt.xlabel('time [s]', fontsize=16)
plt.ylabel('Amplitude', fontsize=16)
plt.ylim([-1.1*max(np.abs(xcorr)), 1.1*max(np.abs(xcorr))])
plt.title('Cross-correlation', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.show()

plt.figure()
plt.plot(t, env, color='black', linewidth=1.2)
plt.axvline(peaks_indexes[0]/fs, 0, env[peaks_indexes[0]]*1.1, color='orange', linestyle='--', linewidth=1.2, label='call')
plt.axvline(peaks_indexes[1]/fs, 0, env[peaks_indexes[1]]*1.1, color='green', linestyle='--', linewidth=1.2, label='echo')
plt.xlabel('time [s]', fontsize=16)
plt.ylabel('Amplitude', fontsize=16)
plt.title('Envelope of the cross-correlation', fontsize=20)
plt.ylim([0, 1.1*max(env)])
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.grid()
plt.show()