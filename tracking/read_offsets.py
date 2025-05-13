import os
import numpy as np
import soundfile as sf
from sonar import sonar
from scipy import signal
from das_v2 import das_filter
from capon import capon_method
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

def pow_two_pad_and_window(vec, show=False):
    window = signal.windows.tukey(len(vec), alpha=0.3)
    windowed_vec = vec * window
    padded_windowed_vec = np.pad(windowed_vec, (0, 2**int(np.ceil(np.log2(len(windowed_vec)))) - len(windowed_vec)))
    if show:
        dur = len(padded_windowed_vec) / fs
        t = np.linspace(0, dur, len(padded_windowed_vec))
        plt.figure()
        plt.plot(t, padded_windowed_vec)
        plt.show()
    return padded_windowed_vec/max(padded_windowed_vec)*0.8

os.chdir(os.path.dirname(os.path.abspath(__file__)))

offsets = np.load('offsets/20250507_18-18-43_offsets.npy')

fs = 176400
dur = 3e-3
hi_freq = 60e3
low_freq = 20e3
output_threshold = -50 # [dB]
distance_threshold = 20 # [cm]

METHOD = 'das' # 'das', 'capon'
if METHOD == 'das':
    spatial_filter = das_filter
elif METHOD == 'capon':
    spatial_filter = capon_method

t_tone = np.linspace(0, dur, int(fs*dur))
chirp = signal.chirp(t_tone, hi_freq, t_tone[-1], low_freq)    
sig = pow_two_pad_and_window(chirp, show=False)

C_AIR = 343
min_distance = 10e-2
discarded_samples = int(np.floor((min_distance*2)/C_AIR*fs))

def update(frame):
    global curr_end
    audio_data = sf.read('20250507_18-18-43.wav', start=curr_end, stop=curr_end + offsets[frame])[0]
    curr_end += offsets[frame]
    dB_rms = 20*np.log10(np.mean(np.std(audio_data, axis=0)))    
    if dB_rms > output_threshold:
        filtered_signals = signal.correlate(audio_data, np.reshape(sig, (-1, 1)), 'same', method='fft')
        roll_filt_sigs = np.roll(filtered_signals, -len(sig)//2, axis=0)
        
        try:
            distance, direct_path, obst_echo = sonar(roll_filt_sigs, discarded_samples, fs)
            distance = distance*100 # [m] to [cm]
            # print('\nDistance: %.1f [cm]' % distance)                             
            # if distance == 0:
            #     print('\nNo Obstacles')
            theta, p = spatial_filter(
                                        roll_filt_sigs[obst_echo - int(5e-4*fs):obst_echo + int(5e-4*fs)], 
                                        fs=fs, nch=roll_filt_sigs.shape[1], d=2.70e-3, 
                                        bw=(low_freq, hi_freq)
                                    )
            p_dB = 10*np.log10(p)
            
            if direct_path != obst_echo:
                doa_index = np.argmax(p_dB)
                theta_hat = theta[doa_index]
                if distance > 0:
                    print('\nDistance: %.1f [cm] | DoA: %.2f [deg]' % (distance, theta_hat))            
                    line.set_ydata(p_dB)
                    ax.set_ylim(min(p_dB), max(p_dB) + 6)
                    vline.set_xdata([np.deg2rad(theta_hat)])

            return line, vline
        except ValueError:
            print('\nNo valid distance or DoA')

global curr_end
curr_end = 0

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.set_title('DaS Filter Output')
# Shift axes by -90 degrees
ax.set_theta_offset(np.pi/2)
# Limit theta between -90 and 90 degrees
ax.set_xlim(-np.pi/2, np.pi/2)
ax.set_ylim(-20, 40)        
ax.grid(False)
line = ax.plot(np.linspace(-np.pi/2, np.pi/2, 73), 0*np.sin(np.linspace(-np.pi/2, np.pi/2, 73)))[0]
vline = ax.axvline(0, 0, 30, color='red', linestyle='--')
ani = FuncAnimation(fig, update,  frames=len(offsets), interval=0, cache_frame_data=False, repeat=False)
plt.show()
