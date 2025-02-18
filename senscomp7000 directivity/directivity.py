import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

if __name__ == '__main__':
    y, fs = sf.read('70deg.wav')
    # y = y[int(0.416*fs):int(0.426*fs)]

    dur = len(y) / fs
    t = np.linspace(0, dur, len(y))

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(t, y)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('70deg.wav')
    plt.subplot(2, 1, 2)
    plt.specgram(y, Fs=fs, NFFT=64, noverlap=32)
    plt.show()