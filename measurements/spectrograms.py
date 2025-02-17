import numpy as np
import librosa
import scipy.signal as signal
import os
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    rec_front, fs = librosa.load('20250214_front1.wav', sr=192e3)
    rec_back, _ = librosa.load('20250214_back1.wav', sr=192e3)
    f_front, t_front, Spec_front = signal.stft(rec_front, fs, nperseg=2048)


    dB_rms_front = 20*np.log10(np.std(rec_front))
    dB_rms_back = 20*np.log10(np.std(rec_back))

    dur_front = len(rec_front)/fs
    time_front = np.linspace(0, dur_front, len(rec_front))
    dur_back = len(rec_back)/fs
    time_back = np.linspace(0, dur_back, len(rec_back))
    
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(time_front, rec_front)
    plt.subplot(2, 2, 2)
    plt.pcolormesh(t_front, f_front, np.abs(Spec_front), vmin=0, vmax=100, shading='gouraud')
    plt.subplot(2, 2, 3)
    plt.plot(time_back, rec_back)
    plt.subplot(2, 2, 4)
    plt.specgram(rec_back, NFFT=2048, Fs=fs)
    plt.tight_layout()
    plt.show()

