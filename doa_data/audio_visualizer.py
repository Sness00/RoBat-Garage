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

x, fs = sf.read('./audio/capon_20250429_18-06-59.wav')

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

plt.figure()
plt.plot(t, x_mono, color='black', linewidth=0.8)
plt.show()

plt.figure()
plt.plot(t, xcorr, color='black', linewidth=0.8)
plt.show()

plt.figure()
plt.plot(t, env, color='black', linewidth=0.8)
plt.show()