# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 11:37:58 2025

@author: gabri
"""
# %% Data
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import librosa
from scipy import signal
from das_v2 import das_filter as das
from capon import capon_method as capon
from music_v2 import music

os.chdir(os.path.dirname(os.path.abspath(__file__)))

M = 16  # number of sensors
L = 0.45  # total length of array [m]
D = L / (M - 1)  # distance between sensors [m]

y, fs = librosa.load('array_recordings.wav', sr=None, mono=False)

x = y.T
METHOD = 'music'
SAVE_VIDEO = False

# %% Processing
K = 1024
big_win = np.reshape(signal.windows.hann(K), (-1, 1))
BIG_HOP = 512
N_frames = (x.shape[0] - K) // BIG_HOP


theta = np.linspace(-90, 90, 145)
p_avg = np.zeros((len(theta), N_frames))
avg_theta = np.zeros(N_frames)

if METHOD == 'das':
    spatial_filter = das
elif METHOD == 'capon':
    spatial_filter = capon
elif METHOD == 'music':
    spatial_filter = music
    
for kk in range(N_frames):
    x_w = x[(kk * BIG_HOP):(kk * BIG_HOP + K)] * big_win
    
    p_avg[:, kk] = spatial_filter(x_w, fs, M, D, (0, fs/2), theta)[1]
    
    avg_theta[kk] = theta[np.argmax(p_avg[:, kk])]

# %% Plot
def update_polar(frame):
    '''
    Frame generator
    '''
    ax.set_ylim(min(p_dB[:, frame]), max(p_dB[:, frame]))
    line.set_ydata(p_dB[:, frame])
    segments.set_xdata(np.array([np.deg2rad(avg_theta[frame])]))
    return line, segments
    # return line

p_dB = 20*np.log10(p_avg)

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.set_title('Delay and Sum Filter Test')
ax.set_theta_direction(1)
ax.set_theta_offset(np.pi / 2)  # Rotate the plot by 90 degrees
ax.set_xlim(-np.pi/2, np.pi/2)
ax.set_ylim(0, 1)
values = np.zeros(theta.shape)
line, = ax.plot(np.deg2rad(theta), values)
segments = ax.axvline(x=np.deg2rad(avg_theta[0]), color='r')
ani = FuncAnimation(fig, update_polar, frames=range(N_frames), interval=100, repeat=False)
if SAVE_VIDEO:
    ani.save('das_test.mp4', writer='ffmpeg', fps=25)
plt.show()
