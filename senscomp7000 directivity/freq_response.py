import os
import numpy as np
from matplotlib import pyplot as plt
import soundfile as sf
from scipy import fftpack

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# use latex for the font in the plot
plt.rcParams['text.usetex'] = True

rec_dir = './gras_20250416/'
sweeps_dir = 'sweeps_1/'
calib_dir = 'calib/'

signals = os.listdir(rec_dir + sweeps_dir)
# Filter out non-WAV files
signals = [f for f in signals if f.endswith('.wav')]

freq_response = []
for signal in signals:
    # Load the signal
    signal_path = os.path.join(rec_dir, sweeps_dir, signal)
    data, fs = sf.read(signal_path)
    # Perform FFT
    # Zero padding to the next power of 2
    N = len(data)
    N = 2**np.ceil(np.log2(N)).astype(int)
    data = np.pad(data, (0, N - len(data)), 'constant')
    T = 1.0 / fs
    yf = fftpack.fft(data)
    xf = np.fft.fftfreq(N, T)[:N//2]
    # append the frequency response
    freq_response.append(20*np.log10(np.sqrt((1/N*np.abs(yf[0:N//2])**2))))

# Convert to numpy array for easier manipulation
freq_response = np.array(freq_response)
# Calculate mean
mean_freq_response_dB = np.mean(freq_response, axis=0)

# Load calibration data
calib_signal = sf.read(os.path.join(rec_dir, calib_dir, 'calibration_tone.wav'))[0]

dB_SPL = 94
# Compute the rms for calibration signal
dB_SPL_to_rms = dB_SPL - 20*np.log10(np.sqrt(np.mean(calib_signal**2))) 

mean_freq_response_dB = mean_freq_response_dB + dB_SPL_to_rms

# plot the frequency response of the calibration signal
plt.figure()
plt.plot(xf, mean_freq_response_dB, color='black')
plt.title('Senscomp 7000 Frequency Response', fontsize=20)
plt.fill_between(xf, mean_freq_response_dB, color='black', alpha=0.1)
plt.xlabel('Frequency (Hz)', fontsize=16)
plt.ylabel('SPL [dB] @ 1 [m] ref 1[kHz] @ 94 [dB SPL]', fontsize=16)
plt.yticks([0, 30, 60, 90], fontsize=16)
plt.xticks([10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000], fontsize=16)
plt.grid()
plt.xlim(15000, 86000)
plt.tight_layout()
plt.show()






    
    