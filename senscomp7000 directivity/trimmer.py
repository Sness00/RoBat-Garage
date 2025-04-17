import soundfile as sf
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import os

def pow_two_pad_and_window(vec, fs, show = True):
    window = signal.windows.tukey(len(vec), alpha=0.2)
    windowed_vec = vec * window
    padded_windowed_vec = np.pad(windowed_vec, (0, 2**int(np.ceil(np.log2(len(windowed_vec)))) - len(windowed_vec)))
    if show:
        dur = len(padded_windowed_vec) / fs
        t = np.linspace(0, dur, len(windowed_vec))
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(t, windowed_vec)
        plt.subplot(2, 1, 2)
        plt.specgram(windowed_vec, NFFT=256, Fs=192e3)
        plt.show()
    return padded_windowed_vec/max(padded_windowed_vec)

def extract_sweep(files, series, fs):
    cut_dir = ['sweeps_1', 'sweeps_2']
    cut_dir = cut_dir[series-1]
    if not os.path.exists(rec_dir + '/' + cut_dir):
        os.makedirs(rec_dir + '/' + cut_dir)

    
    dur = 6e-3
    hi_freq = 95e3
    low_freq = 15e3

    t_tone = np.linspace(0, dur, int(fs*dur))
    chirp = signal.chirp(t_tone, hi_freq, t_tone[-1], low_freq)
    sig = pow_two_pad_and_window(chirp, fs, show=False)
    
    for file_name in files:
        print(file_name)
        x = sf.read(rec_dir + '/' + file_name)[0]
        xcorr = signal.correlate(x, sig, mode='same')
        xcorr_rolled = np.roll(xcorr, -len(sig)//2)
        envelope = np.abs(signal.hilbert(xcorr_rolled))
        idxs = signal.find_peaks(envelope, prominence=0.5, distance=int(60e-3*fs))[0]
        print(idxs.shape)
        for n, i in enumerate(idxs[5*(series-1):5*series]):
            x_trimmed = x[i-384:i + int(dur*fs) + 384]
            sf.write(rec_dir + '/' + cut_dir + '/' +
                    file_name[0:3] + 'deg_' + str(n+1) + '.wav', x_trimmed, int(fs))
            if int(file_name[0:3]) > 0 and int(file_name[0:3]) < 180:
                x_symm = x_trimmed
                file_name_symm = str(360 - int(file_name[0:3])) + 'deg' + '_' + str(n+1) + '.wav'
                sf.write(rec_dir + '/' + cut_dir + '/' + file_name_symm, x_symm, int(fs))

def extract_noise_floor(files, fs):
    cut_dir = 'noise_floor'
    if not os.path.exists(rec_dir + '/' + cut_dir):
        os.makedirs(rec_dir + '/' + cut_dir)
    for file_name in files:
        print(file_name)
        x = sf.read(rec_dir + '/' + file_name, start=int(fs), frames=int(fs*0.01))[0]
        sf.write(rec_dir + '/' + cut_dir + '/' + file_name, x, int(fs))
        if int(file_name[0:3]) > 0 and int(file_name[0:3]) < 180:
                x_symm = x
                file_name_symm = str(360 - int(file_name[0:3])) + 'deg' + '.wav'
                sf.write(rec_dir + '/' + cut_dir + '/' + file_name_symm, x_symm, int(fs))

if __name__ == '__main__':

    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)    

    fs = 192e3

    rec_dir = './sanken_20250416/' # Directory where the recordings are stored

    audio_files = os.listdir(rec_dir) # List all files in the sweeps directory
    audio_files = [f for f in audio_files if f.endswith('.wav')] # Keep only the files with 3 digits and .wav extension
    series = 1
    extract_sweep(audio_files, series, fs)
    extract_noise_floor(audio_files, fs)