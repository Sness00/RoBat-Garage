import os
import numpy as np
import soundfile as sf
from scipy import signal
from matplotlib import pyplot as plt
from scipy.signal import ShortTimeFFT
from scipy.signal import windows
from matplotlib.image import NonUniformImage

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

plt.rcParams.update({
"text.usetex": True,
"font.family": "serif",
"font.serif": ["Computer Modern Roman"],
"text.latex.preamble": r"""
\usepackage{lmodern}
\renewcommand{\rmdefault}{cmr}
\renewcommand{\sfdefault}{cmss}
\renewcommand{\ttdefault}{cmtt}
""",
    "font.size": 16,           # Set default font size
    "axes.labelsize": 16,      # Axis label font size
    "xtick.labelsize": 16,     # X tick label font size
    "ytick.labelsize": 16,     # Y tick label font size
    "legend.fontsize": 16,     # Legend font size
    "axes.titlesize": 16       # Title font size
})

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

win = windows.hann(128)
SFT = ShortTimeFFT(win, 64, fs, fft_mode='onesided', scale_to='magnitude', mfft=256)
ss = SFT.spectrogram(env)/128**2
t2 = np.linspace(0, len(env)/fs, SFT.p_num(len(env)))
freq2 = np.linspace(0, (fs/2 - fs/SFT.f_pts), SFT.f_pts)

fig = plt.figure()
ax2 = fig.add_subplot()
im2 = NonUniformImage(ax2, interpolation='bilinear', cmap='inferno')
im2.set_data(t2, freq2, 10*np.log10(ss))
ax2.add_image(im2)
# ax2.set_xlim(0, 0.003)
ax2.set_ylim(15e3, 65e3)
ax2.set_xlabel('Time [ms]')
# ax2.set_xticks(ticks=[0, .001, .002, .003], labels=['0', '1', '2', '3'])
ax2.set_ylabel('Frequency [kHz]')
ax2.set_yticks(ticks=[20000, 30000, 40000, 50000, 60000],
           labels=['20', '30', '40', '50', '60'], fontsize=16)
# ax2.set_title('Spectrogram of the call signal')
cbar2 = fig.colorbar(im2, ax=ax2, orientation='vertical')
cbar2.set_label('Magnitude [dB]')
plt.show()

# plt.figure(figsize=(12, 4))
# plt.plot(t, x_mono, color='black', linewidth=0.8)
# plt.xlabel('Time [s]', fontsize=16)
# plt.ylabel('Amplitude', fontsize=16)
# plt.ylim([-1.1*max(np.abs(x_mono)), 1.1*max(np.abs(x_mono))])
# plt.xticks(fontsize=16)
# plt.xlim([0.01, 0.02])
# plt.yticks(fontsize=16)
# plt.title('Recorded audio', fontsize=20)
# plt.tight_layout()
# plt.grid()
# # plt.show()
# plt.savefig('audio', dpi=600, transparent=True)

# plt.figure(figsize=(12, 4))
# plt.plot(t, xcorr, color='black', linewidth=0.8)
# plt.xlabel('Time [s]', fontsize=16)
# plt.ylabel('Amplitude', fontsize=16)
# plt.ylim([-1.1*max(np.abs(xcorr)), 1.1*max(np.abs(xcorr))])
# plt.title('Cross-correlation', fontsize=20)
# plt.xticks(fontsize=16)
# plt.xlim([0.01, 0.02])
# plt.yticks(fontsize=16)
# plt.tight_layout()
# plt.grid()
# plt.savefig('cc', dpi=600, transparent=True)
# # plt.show()

# plt.figure(figsize=(12, 4))
# plt.plot(t, env, color='black', linewidth=1.2)
# plt.axvline(peaks_indexes[0]/fs, 0, env[peaks_indexes[0]]*1.1, color='orange', linestyle='--', linewidth=1.2, label='call')
# plt.axvline(peaks_indexes[1]/fs, 0, env[peaks_indexes[1]]*1.1, color='green', linestyle='--', linewidth=1.2, label='echo')
# plt.xlabel('Time [s]', fontsize=16)
# plt.xlim([0.01, 0.02])
# plt.ylabel('Amplitude', fontsize=16)
# plt.title('Envelope of the cross-correlation', fontsize=20)
# plt.ylim([0, 1.1*max(env)])
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.legend(fontsize=16)
# plt.grid()
# plt.tight_layout()
# plt.savefig('env', dpi=600, transparent=True)
# # plt.show()
