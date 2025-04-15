import os
import soundfile
import numpy as np
from matplotlib import pyplot as plt
from scipy import fft

# Load audio files, then plot them in a 6x6 grid
SIG_DIR = "./sanken_CO-100K/sweeps_2/"  # Directory containing the audio files
signal_files = os.listdir(SIG_DIR)  # List all files in the sweeps directory

NOISE_DIR = "./sanken_CO-100K/noise_floor/"  # Directory containing the audio files
noise_files = os.listdir(NOISE_DIR)

snrs = []
for i in np.arange(36):
    noise = soundfile.read(NOISE_DIR + noise_files[i])[0]
    snr = 0
    for j in np.arange(5*i, 5*(i + 1)):
        signal = soundfile.read(SIG_DIR + signal_files[j])[0]
        # Compute the SNR
        snr += 10 * np.log10(np.mean(signal**2) / np.mean(noise**2))
    snrs.append(snr / 5)
print(np.array(snrs).shape)
plt.figure()
plt.stem(np.linspace(0, 350, 36), snrs, markerfmt="o", basefmt=" ")
plt.title("SNR - Senscomp Series 7000 Transducer")
plt.xlabel("Angle [deg]")
plt.ylabel("SNR [dB]")
plt.grid()
plt.show()