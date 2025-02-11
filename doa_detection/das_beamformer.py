import numpy as np
from scipy.signal import stft

def das_filter(y, fs, nch, theta, c, d):
    t_spec_axis, f_spec_axis, spectrum = stft(y, fs=fs, nperseg=256, noverlap=255, axis=0)
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