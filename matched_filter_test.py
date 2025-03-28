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

attenuation = 11.5
print('Echo attenuation: %.0f [dB]' % (20*np.log10(1/attenuation)))
x = (x1_win/attenuation + x2_aligned)/2
x_padded = np.pad(x, (int(fs*0.01), int(fs*0.005)), 'constant')

plt.figure()
plt.title('Echo and Masking Call')
plt.plot(t1, x1_win/attenuation, label='echo')
plt.plot(t1, x2_aligned, label='masking call')
plt.legend()
plt.xlabel('time [s]')
plt.ylabel('amplitude')
plt.grid()
plt.show()

plt.figure()
plt.title('Spectrogram of the overlapped signals')
plt.specgram(x, NFFT=128, Fs=fs, noverlap=64, cmap='inferno', sides='onesided', scale_by_freq=False)
plt.xlabel('time [s]')
plt.ylabel('frequency [Hz]')
plt.show()

mf = np.roll(signal.correlate(x_padded, x1_win, 'same', method='fft'), -len(x1_win)//2)
mf_envelope = np.abs(signal.hilbert(mf))

plt.figure()
plt.suptitle('Matched Filter')

plt.subplot(2, 1, 1)
plt.plot(x_padded)
plt.axvline(int(fs*0.01), -1, 1, color='r', linestyle='--')
plt.text(int(fs*0.01) + 100, 0.25, 'echo beginning at sample %d' % int(fs*0.01), color='r')
plt.xlabel('samples')
plt.ylabel('amplitude')
plt.grid()
plt.title('Echo and Masking Call with padding')

plt.subplot(2, 1, 2)
plt.plot(mf_envelope)
plt.xlabel('samples')
plt.ylabel('amplitude')
plt.grid()
plt.title('Matched Filter Output')
plt.axvline(np.argmax(mf_envelope), 0, max(mf_envelope), color='r', linestyle='--')
plt.text(np.argmax(mf_envelope) + 100, max(mf_envelope)/2, 'cc peak at sample %d' % np.argmax(mf_envelope) , color='r')

plt.tight_layout()
plt.show()
