import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import scipy.signal as signal
from matplotlib.animation import FuncAnimation
from das_v2 import das_filter
from music_v2 import music
import time
# Define DAS filter function
# Constants
c = 343.0  # speed of sound
fs = 192e3  # sampling frequency
mic_spacing = 2.70e-3
channels = 8
block_size = 2048
freq_range = [15e3, 60e3]


S = sd.InputStream(samplerate=fs, blocksize=block_size, device=3, channels=channels, latency='low')

def initialization():
    try:
        S.start()
        print("Stream started")

    except KeyboardInterrupt:
        print("\nStream stopped by user")
    except Exception as e:
        print(f"An error occurred: {e}")

def update_polar(frame):
    global rec
    in_sig, status = S.read(S.blocksize)
    theta, spatial_resp = das_filter(in_sig, fs, channels, mic_spacing, freq_range)    
    # theta, spatial_resp = music(in_sig, fs, channels, mic_spacing, freq_range)
    #spatial_resp = 10 * np.log10(spatial_resp)
    #print(spatial_resp)
    # spatial_resp = (spatial_resp - spatial_resp.min()) / (spatial_resp.max() - spatial_resp.min())
    #window = signal.windows.tukey(37,alpha=1)
    #spatial_resp = spatial_resp * window
    p_dB = 20*np.log10(spatial_resp)
    ax.set_ylim(min(p_dB), max(p_dB))
    line.set_ydata(p_dB)
    return line,

initialization()
memory, rec = [], []
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.set_theta_direction(1)
ax.set_theta_offset(np.pi / 2)  # Rotate the plot by 90 degrees
theta = np.linspace(-np.pi/2, np.pi/2, 73)
ax.set_thetamin(-90)
ax.set_thetamax(90)
ax.set_xticks(np.pi/180. * np.linspace(-90, 90, 19), labels=np.arange(-90, 91, 10))
#ax.set_yticks([1e-11, 1e-5])
values = np.random.rand(73)
#values = np.zeros(37)
line, = ax.plot(theta, values)
ani = FuncAnimation(fig, update_polar, frames=range(73), blit=False, interval=10)
plt.show()
