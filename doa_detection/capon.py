# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 17:07:50 2025

@author: gabri
"""

import numpy as np
from scipy.signal import stft

def capon_method(y, fs, nch, d, bw, theta=np.linspace(-90, 90, 73), c=343):    
    """
    Simple multiband Capon Method spatial filter implementation.
    
    Parameters:
        y: mic array signals.
        
        fs: sampling rate.
        
        nch: number of mics in the array.
        
        d: mic spacing.
        
        bw: (low freq, high freq).
        
        theta: angle vector.
        
        c: sound speed.

    Returns:
        theta: angle vector.
        
        mag_p: average spatial energy distribution estimation.
    """
    win_len = 128
    f_spec_axis, _, spectrum = stft(y, fs=fs, window=np.ones((win_len, )), nperseg=win_len, noverlap=win_len-1, axis=0)
    bands = f_spec_axis[(f_spec_axis >= bw[0]) & (f_spec_axis <= bw[1])]
    p = np.zeros_like(theta, dtype=complex)
    
    for f_c in bands:
        w_s = (2*np.pi*f_c*d*np.sin(np.deg2rad(theta))/c)        
        a = np.exp(np.outer(np.linspace(0, nch-1, nch), -1j*w_s))
        a_H = a.T.conj()     
        spec = spectrum[f_spec_axis == f_c, :, :].squeeze()
        cov_est = np.cov(spec, bias=True)
        inv_cov_est = np.linalg.inv(cov_est)
        for i, _ in enumerate(theta):
          p[i] += 1/(a_H[i, :] @ inv_cov_est @ a[:, i])
    
    mag_p = np.abs(p)/len(bands)
        
    return theta, mag_p