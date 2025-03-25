import numpy as np
from scipy.signal import stft
from matplotlib import pyplot as plt

def das_filter(y, fs, nch, d, bw, theta=np.linspace(-90, 90, 73), c=343, wlen=64, show=False):    
    """
    Simple multiband Delay-and-Sum spatial filter implementation.
    
    Parameters:
        
        y: mic array signals
        
        fs: sampling rate
        
        nch: number of mics in the array
        
        d: mic spacing
        
        bw: (low freq, high freq)
        
        theta: angle vector
        
        c: sound speed
        
        wlen: window length for stft
        
        show: plot the pseudospectrum for each band
        
    Returns:
        
        theta: angle axis
        
        mag_p: magnitude of average spatial energy distribution estimation.
    """
    f_spec_axis, _, spectrum = stft(y, fs=fs, window=np.ones((wlen, )), nperseg=wlen, noverlap=wlen-1, axis=0)
    bands = f_spec_axis[(f_spec_axis >= bw[0]) & (f_spec_axis <= bw[1])]
    p = np.zeros_like(theta, dtype=complex)
    p_i = np.zeros((len(theta), 1), dtype=complex)

    for f_c in bands:
        w_s = (2*np.pi*f_c*d*np.sin(np.deg2rad(theta))/c)        
        a = np.exp(np.outer(np.linspace(0, nch-1, nch), -1j*w_s))
        a_H = a.T.conj()     
        spec = spectrum[f_spec_axis == f_c, :, :].squeeze()
        cov_est = np.cov(spec, bias=True)
        
        for i, _ in enumerate(theta):
          
          p_i[i] = a_H[i, :] @ cov_est @ a[:, i]/(nch**2)
          p[i] += p_i[i]
        if show:
          plt.figure()
          plt.polar(np.deg2rad(theta), 20*np.log10(np.abs(p_i)))
          plt.xlim((-np.pi/2, np.pi/2))
          plt.ylim((-90, 30))
          plt.title(str(f_c))
          plt.show()
    
    mag_p = np.abs(p)/len(bands)
        
    return theta, mag_p