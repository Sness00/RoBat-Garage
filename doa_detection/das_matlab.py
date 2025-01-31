import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal.windows import hann

def my_stft(signal, fs, nch, hop_size):
    w_len = 256 
    nfft = 256
    s_len = signal.shape[1]
    n_frames = (s_len - w_len) // hop_size

    y = np.zeros((nfft, n_frames, nch), dtype=complex)
    for i in range(n_frames):
        i_samples = slice(i * hop_size, i * hop_size + w_len)
        yi = signal[:, i_samples]
        Yi = np.fft.fft(yi, nfft, axis=1)
        y[:, i, :] = Yi.T

    t = np.arange(n_frames) * hop_size / fs
    f = np.arange(nfft // 2 + 1) * fs / nfft
    spec = y[:nfft // 2 + 1, :, :]

    return spec, t, f

def das_filter(y, fs, nch, theta, c, d):
    spectrum, t_spec_axis, f_spec_axis = my_stft(y, fs, nch, 1)
    bands = f_spec_axis[2::4]
    a = np.zeros((nch, len(theta), len(bands)), dtype=complex)
    cov_est = np.zeros((nch, nch, len(bands)), dtype=complex)
    for f_c in bands:
        a[:, :, bands == f_c] = np.expand_dims(np.exp(-1j * 2 * np.pi * f_c * d * np.sin(np.deg2rad(theta)) * np.arange(nch)[:, None] / c), 2)
        for ii in range(len(t_spec_axis)):
            spec = spectrum[f_spec_axis == f_c, ii, :].squeeze()
            cov_est[:, :, bands == f_c] += np.expand_dims(np.outer(spec, spec.conj().T), 2) / len(t_spec_axis)

    p = np.zeros((len(bands), len(theta)))
    for ii in range(len(bands)):
        for jj in range(len(theta)):
            p[ii, jj] = np.abs(a[:, jj, ii].conj().T @ cov_est[:, :, ii] @ a[:, jj, ii]) / nch**2
    
    avg_pseudo_spec = np.sum(p, axis=0) / len(bands)
    return avg_pseudo_spec

# DATA AND INITIALIZATION
M = 16  # number of sensors
L = 0.45  # total length of array [m]
c = 343  # measured speed of sound [m/s]
d = L / (M - 1)  # distance between sensors [m]

# d < lambda/2
lambda_min = 2 * d
f_max = c / lambda_min  # anti-aliasing condition

fs, y = wavfile.read('array_recordings.wav')

# PROCESSING
K = 1024
big_win = hann(K)
big_hop = 512
N_frames = (len(y) - K) // big_hop

theta = np.arange(-90, 91, 1)
p_avg = np.zeros((N_frames, len(theta)))
avg_theta = np.zeros(N_frames)

for kk in range(N_frames):
    y_w = y[(kk * big_hop):(kk * big_hop + K), :].T * big_win
    
    p_avg[kk, :] = das_filter(y_w, fs, M, theta, c, d)
    # p_avg[kk, :] = capon(y_w, fs, M, theta, c, d)
    
    avg_theta[kk] = theta[np.argmax(np.abs(p_avg[kk, :]))]

# PSEUDO-SPECTRUM OVER TIME
abs_p_avg = np.abs(p_avg)
abs_p_avg /= np.max(abs_p_avg, axis=1, keepdims=True)
tax, theax = np.meshgrid(theta, np.arange(N_frames) * big_hop / fs)
plt.figure()
plt.subplot(2, 1, 1)
plt.pcolormesh(theax, tax, abs_p_avg, shading='auto')
plt.yticks([-90, -60, -30, 0, 30, 60, 90])
plt.ylim([-90, 90])
plt.xticks(np.arange(0, 15, 2))
plt.xlim([0, 14.6])
plt.xlabel('time [s]')
plt.ylabel('θ [deg]')
plt.title('Normalized pseudo-spectrum over time')
plt.colorbar()

plt.subplot(2, 1, 2)
plt.plot(np.arange(N_frames) * big_hop / fs, avg_theta, linewidth=1.2)
plt.xticks(np.arange(0, 15, 2))
plt.xlim([0, 14.6])
plt.yticks([-90, -60, -30, 0, 30, 60, 90])
plt.ylim([-90, 90])
plt.xlabel('time [s]')
plt.ylabel('θ_{dir} [deg]')
plt.title('Estimated DOA')
plt.grid(True)
plt.tight_layout()
plt.show()

# Signals plot
# time = np.arange(0, len(y)) / fs
# plt.figure()
# plt.suptitle('Normalized array signals')
# for ii in range(16):
#     plt.subplot(4, 4, ii + 1)
#     plt.plot(time, y[:, ii] / np.max(np.partition(y, -1, axis=0)[-1, :]))
#     plt.yticks([-1, -0.5, 0, 0.5, 1])
#     plt.ylim([-1.1, 1.1])
#     plt.xlabel('time [s]')
#     plt.xticks([2, 6, 10, 14])
#     plt.title(f'Microphone {ii + 1}')
#     plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Spectrograms plot
# spectrum, t_spec_axis, f_spec_axis = my_stft(y.T, fs, M, 256)

# ff, tt = np.meshgrid(t_spec_axis, f_spec_axis)

# plt.figure()
# for ii in range(M):
#     plt.subplot(4, 4, ii + 1)
#     plt.pcolormesh(ff, tt, np.abs(spectrum[:, :, ii]), shading='auto')
#     plt.title(f'Microphone {ii + 1}')
#     plt.colorbar()
#     plt.xticks([2, 6, 10, 14])
#     plt.xlim([0, 14.7])
#     plt.xlabel('time [s]')
#     plt.ylabel('frequency [Hz]')
# plt.suptitle('Spectrograms of the signals')
# plt.tight_layout()
# plt.show()

