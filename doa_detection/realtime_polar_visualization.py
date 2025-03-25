import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from matplotlib.animation import FuncAnimation
from das_v2 import das_filter as das
from music_v2 import music
from capon import capon_method as capon

# Define DAS filter function
# Constants
c = 343.0  # speed of sound
fs = 44.1e3  # sampling frequency
mic_spacing = 2.70e-3
channels = 8
block_size = 0
freq_range = (1000, 20000)

METHOD = 'das'

if METHOD == 'das':
    spatial_filter = das
elif METHOD == 'capon':
    spatial_filter = capon
elif METHOD == 'music':
    spatial_filter = music

S = sd.InputStream(samplerate=fs, blocksize=block_size, device=3, channels=channels, latency='low')

def update_polar(frame):
    global rec
    global S
    in_sig, status = S.read(S.blocksize)
    print(status)
    theta, spatial_resp = spatial_filter(in_sig, fs, channels, mic_spacing, freq_range, theta=np.linspace(90, -90, 73), c=343, wlen=64)
    #spatial_resp = 10 * np.log10(spatial_resp)
    #print(spatial_resp)
    spatial_resp = (spatial_resp - spatial_resp.min()) / (spatial_resp.max() - spatial_resp.min())
    #window = signal.windows.tukey(37,alpha=1)
    #spatial_resp = spatial_resp * window
    line.set_ydata(spatial_resp)
    return line

memory, rec = [], []
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.set_theta_direction(-1)
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
with S:
    print("Stream started")
    plt.show()
