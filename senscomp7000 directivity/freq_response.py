import os
import numpy as np
from matplotlib import pyplot as plt
import soundfile as sf
from scipy import fft
from scipy.signal import butter, sosfilt, windows

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def bandpass_rms(signal, fs, lowcut=950, highcut=1050, order=4):
    """
    Apply a bandpass filter and compute RMS.
    
    Parameters:
    - signal: The calibration signal (1D numpy array)
    - fs: Sampling frequency
    - lowcut, highcut: Band edges in Hz
    - order: Filter order (default 4)

    Returns:
    - RMS of the filtered signal
    """
    sos = butter(order, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')
    filtered = sosfilt(sos, signal)
    rms = np.sqrt(np.mean(filtered**2))
    return rms

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

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
    yf = fft.rfft(data) * 2 / N
    xf = fft.rfftfreq(N, T)
    # append the frequency response
    # yf[1:] = yf[1:] * 2  # Double the amplitude for positive frequencies
    freq_response.append(moving_average(np.abs(yf), 32)/np.sqrt(2))

# Convert to numpy array for easier manipulation
freq_response = np.array(freq_response)
# Calculate mean
mean_freq_response_dB = 20*np.log10(np.mean(freq_response, axis=0))

# Load calibration data
calib_signal = sf.read(os.path.join(rec_dir, calib_dir, 'calibration_tone.wav'))[0]
dB_SPL = 94
# Compute the rms for calibration signal
dB_SPL_to_rms = dB_SPL - 20*np.log10(bandpass_rms(calib_signal, fs, lowcut=950, highcut=1050, order=4))

mean_freq_response_dB = mean_freq_response_dB + dB_SPL_to_rms

# plot the frequency response of the calibration signal
plt.figure()
plt.plot(xf, mean_freq_response_dB, color='black')
plt.title('Senscomp Series 7000 Frequency Response', fontsize=20)
plt.fill_between(xf, mean_freq_response_dB, color='black', alpha=0.1)
plt.xlabel('Frequency [Hz]', fontsize=16)
plt.ylabel('SPL [dB] @ 1 [m] ref 20[$\\mu$Pa]', fontsize=16)
plt.yticks([0, 20, 40, 60, 80], fontsize=16)
plt.xticks([10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000], fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.xlim(15000, 96000)
plt.tight_layout()
plt.show()
