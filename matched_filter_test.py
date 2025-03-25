from matplotlib import pyplot as plt
import numpy as np
from scipy import signal

fs = 192e3

dur1 = 20e-3
t1 = np.linspace(0, dur1, int(fs*dur1))
x1 = signal.chirp(t1, 80e3, t1[-1], 20e3) 
win1 = signal.windows.tukey(len(x1), alpha=0.2)
x1_win = x1*win1

dur2 = 10e-3
t2 = np.linspace(0, dur2, int(fs*dur2))
x2 = signal.chirp(t2, 80e3, t2[-1], 20e3)
win2 = signal.windows.tukey(len(x2), alpha=0.2)
x2_win = x2*win2
x2_aligned = np.concatenate((np.zeros(int(fs*0.35*dur2)), x2_win, np.zeros(int(fs*0.65*dur2))))

attenuation = 16
x = (x1_win/attenuation + x2_aligned)/2

plt.figure()
plt.plot(t1, x1_win/attenuation)
plt.plot(t1, x2_aligned)
plt.show()

plt.figure()
plt.specgram(x, NFFT=128, Fs=fs, noverlap=64, cmap='inferno')
plt.show()

mf = np.roll(signal.correlate(x, x1_win, 'same', method='fft'), -len(x1_win)//2)

plt.figure()
plt.plot(t1, mf)
plt.axvline(np.argmax(np.abs(mf))/fs, -max(np.abs(mf)), max(np.abs(mf)), color='r', linestyle='--')
plt.show()
