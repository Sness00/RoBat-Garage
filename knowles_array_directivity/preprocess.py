# Extract signals
import os
import numpy as np
import soundfile as sf
from scipy import signal
import matplotlib.pyplot as plt

os.chdir(os.path.abspath(os.path.dirname(__file__)))
CUT_DIR = 'gras'
file_name = 'gras_recording.wav'
durn = 3e-3  # duration of the chirp in seconds
fs = 192000  # sampling frequency in Hz
t = np.linspace(0, durn, int(fs*durn))
start_f, end_f = 15e3, 95e3
sweep = signal.chirp(t, start_f, t[-1], end_f)
sweep *= signal.windows.tukey(sweep.size, 0.2)
sweep *= 0.8
sig = sweep

for i in np.arange(1):
    save_dir = os.path.join('calibration', CUT_DIR + str(i+1))
    print(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)       
    x = sf.read(file_name)[0]
    # x = x[:, i]            
    xcorr = signal.correlate(x, sig, mode='same')
    xcorr_rolled = np.roll(xcorr, -len(sig)//2)
    envelope = np.abs(signal.hilbert(xcorr_rolled))
    idxs = signal.find_peaks(envelope, prominence=0.5, distance=int(30e-3*fs))[0]
    print(f'Found {len(idxs)} peaks in {file_name} for mic {i+1}')
    for n, idx in enumerate(idxs):
        print(idx/fs)
        x_trimmed = x[idx-384:idx + int(durn*fs) + 384]
        sf.write(save_dir + '/' + str(n) + '_' + file_name, x_trimmed, int(fs))

# # Extract noise floor
# audio_files = [f for f in os.listdir(DIR) if f.endswith('.wav')]
# for i in np.arange(8):
#     save_dir = os.path.join('noise_floor', CUT_DIR + str(i+1))
#     print(save_dir)
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     for f in audio_files:
#         x = sf.read(os.path.join(DIR, f), start=int(0.01*fs), frames=int(fs*0.01))[0]
#         x = x[:, i]
#         sf.write(save_dir + '/' + f[0:3] + 'deg' + '.wav', x, int(fs))