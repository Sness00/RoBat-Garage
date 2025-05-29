"""

"""
# %% Libraries and files

import os
import soundfile
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

plt.rcParams['text.usetex'] = True

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
in_bands = True
# Load audio files, then plot them in a 6x6 grid
DIR = "./sanken_20250416/sweeps_1/"  # Directory containing the audio files
audio_files = os.listdir(DIR)  # List all files in the sweeps directory

# %% Radiance computation
NFFT = 2048
radiances = []
for i in np.arange(5):
    channels = []
    for j in np.arange(i, len(audio_files), 5):
        audio, fs = soundfile.read(DIR + audio_files[j])
        channels.append(audio)
    channels = np.array(channels)

    Channels = fft.fft(channels, n=NFFT, axis=1)
    Channels_uni = Channels[:, 0:NFFT//2]
    freqs = fft.fftfreq(NFFT, 1 / fs)
    freqs = freqs[0:NFFT//2]
    R = 1
    radiance = 4 * np.pi * R * np.abs(Channels_uni)
    for j in range(len(radiance)):
        radiance[j, :] = moving_average(radiance[j, :], 32)
    radiances.append(radiance)

radiances = np.array(radiances)
mean_radiance = np.mean(radiances, axis=0)

theta = np.linspace(0, 350, 36)
theta = np.append(theta, theta[0])

# %% Radiance display at multiple frequencies
if in_bands:
    central_freq = np.array([20e3, 30e3, 40e3, 50e3, 60e3, 70e3, 80e3, 90e3])
    BW = 2e3

    linestyles = ["-", "--", "-.", ":"]

    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "polar"})
    plt.suptitle("Radiance Pattern - Senscomp Series 7000 Transducer", fontsize=20)
    i = 3
    for fc in central_freq[0:4]:
        rad_patt = np.mean(
            mean_radiance[:, (freqs < fc + BW) & (freqs > fc - BW)], axis=1
        )
        rad_patt_norm = rad_patt / np.max(rad_patt)
        rad_patt_norm_dB = 20 * np.log10(rad_patt_norm)
        rad_patt_norm_dB = np.append(rad_patt_norm_dB, rad_patt_norm_dB[0])
        ax1.plot(
            np.deg2rad(theta),
            rad_patt_norm_dB,
            label=str(fc)[0:2] + " [kHz]",
            linestyle=linestyles[i],
            linewidth=2
        )
        i -= 1
    ax1.legend(loc="upper right", bbox_to_anchor=(1.3, 1.2), fontsize=16)
    # offset polar axes by -90 degrees
    ax1.set_theta_offset(np.pi / 2)
    # set theta direction to clockwise
    ax1.set_theta_direction(-1)
    # more theta ticks
    ax1.set_xticks(np.linspace(0, 2 * np.pi, 18, endpoint=False))
    ax1.set_yticks(np.linspace(-40, 0, 5))
    for label in ax1.get_yticklabels():
        label.set_fontsize(16)
    for label in ax1.get_xticklabels():
        label.set_fontsize(16)
    ax1.set_ylabel("dB", {'fontsize': 16}, labelpad=20)
    ax1.yaxis.label.set_rotation(0)
    # less radial ticks
    ax1.set_rlabel_position(100)
    ax1.set_rlabel_position(-90)

    i = 3
    ax2.plot([])
    ax2.plot([])
    ax2.plot([])
    ax2.plot([])

    for fc in central_freq[4:8]:
        rad_patt = np.mean(
            mean_radiance[:, (freqs < fc + BW) & (freqs > fc - BW)], axis=1
        )
        rad_patt_norm = rad_patt / np.max(rad_patt)
        rad_patt_norm_dB = 20 * np.log10(rad_patt_norm)
        rad_patt_norm_dB = np.append(rad_patt_norm_dB, rad_patt_norm_dB[0])
        
        ax2.plot(
            np.deg2rad(theta),
            rad_patt_norm_dB,
            label=str(fc)[0:2] + " [kHz]",
            linestyle=linestyles[i],
            linewidth=2
        )
        i -= 1
    ax2.legend(loc="upper right", bbox_to_anchor=(1.3, 1.2), fontsize=16)
    # offset polar axes by -90 degrees
    ax2.set_theta_offset(np.pi / 2)
    # set theta direction to clockwise
    ax2.set_theta_direction(-1)
    # more theta ticks
    ax2.set_xticks(np.linspace(0, 2 * np.pi, 18, endpoint=False))
    ax2.set_yticks(np.linspace(-40, 0, 5))
    for label in ax2.get_xticklabels():
        label.set_fontsize(16)  # Set to desired font size
    for label in ax2.get_yticklabels():
        label.set_fontsize(16)
    # less radial ticks
    ax2.set_ylabel("dB", {'fontsize': 16}, labelpad=20)
    ax2.yaxis.label.set_rotation(0)
    ax2.set_rlabel_position(100)
    ax2.set_rlabel_position(-90)

    plt.tight_layout()
    plt.show()
# %%
# fig.savefig('radiation', transparent=True)
# %% Mean radiance pattern display

rad_patt = np.mean(radiance, axis=1)
rad_patt_norm = rad_patt / np.max(rad_patt)
rad_patt_norm_dB = 20 * np.log10(rad_patt_norm)
rad_patt_norm_dB = np.append(rad_patt_norm_dB, rad_patt_norm_dB[0])

fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
ax.plot(np.deg2rad(theta), rad_patt_norm_dB, linewidth=2, color='black')
# offset polar axes by -90 degrees
ax.set_theta_offset(np.pi / 2)
# set theta direction to clockwise
ax.set_theta_direction(-1)
# more theta ticks
ax.set_xticks(np.linspace(0, 2 * np.pi, 18, endpoint=False))
for label in ax.get_xticklabels():
    label.set_fontsize(16)  # Set to desired font size

ax.set_ylabel("dB", {'fontsize': 16}, labelpad=20)
ax.yaxis.label.set_rotation(0)
ax.yaxis.label.set_ha('right')
# less radial ticks
ax.set_yticks(np.linspace(-40, 0, 5))
for label in ax.get_yticklabels():
    label.set_fontsize(16)  # Set to desired font size
ax.set_rlabel_position(-90)
ax.set_title(
    "Senscomp Series 7000 Transducer Mean Radiance Pattern 15[kHz] - 95[kHz]",
    {'fontsize': 20}
)
plt.tight_layout()
plt.show()

