"""

"""
# %% Libraries and files

import os
import soundfile
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Load audio files, then plot them in a 6x6 grid
DIR = "./cut_sweeps/"  # Directory containing the audio files
audio_files = os.listdir(DIR)  # List all files in the sweeps directory

# %% Plot of the collected data

fig, axs = plt.subplots(6, 6, figsize=(20, 20))

for i in range(6):
    for j in range(6):
        # Load audio file
        audio, fs = soundfile.read(DIR + audio_files[i * 6 + j])
        # Plot audio file
        axs[i, j].plot(np.linspace(0, len(audio) / fs, len(audio)), audio)
        axs[i, j].set_title(audio_files[i * 6 + j])
        axs[i, j].set_xlabel("Time (s)")
        axs[i, j].set_ylabel("Amplitude")
        # Shared x and y axes
        axs[i, j].sharex(axs[0, 0])
        axs[i, j].sharey(axs[0, 0])

plt.tight_layout()
plt.show()

# %% Radiance computation

channels = []
for i in np.arange(len(audio_files)):
    audio, fs = soundfile.read(DIR + audio_files[i])
    if audio.shape[0] > 1919:
        audio = audio[0:1919]
    channels.append(audio)
channels = np.array(channels)

Channels = fft.fft(channels, n=2048, axis=1)
Channels_uni = Channels[:, 0:1024]
freqs = fft.fftfreq(2048, 1 / fs)
freqs = freqs[0:1024]
R = 1
radiance = 4 * np.pi * R * np.abs(Channels_uni)
theta = np.linspace(0, 350, 36)
theta = np.append(theta, theta[0])

# %% Radiance display at multiple frequencies

central_freq = np.array([20e3, 30e3, 40e3, 50e3, 60e3, 70e3, 80e3, 90e3])
BW = 2e3

linestyles = ["-", "--", "-.", ":"]

fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "polar"})
plt.suptitle("Radiance Pattern - Senscomp Series 7000 Transducer", fontsize=35)
i = 3
for fc in central_freq[0:4]:
    rad_patt = np.mean(
        radiance[:, (freqs < fc + BW) & (freqs > fc - BW)], axis=1
    )
    rad_patt_norm = rad_patt / np.max(rad_patt)
    rad_patt_norm_dB = 20 * np.log10(rad_patt_norm)
    rad_patt_norm_dB = np.append(rad_patt_norm_dB, rad_patt_norm_dB[0])
    ax1.plot(
        np.deg2rad(theta),
        rad_patt_norm_dB,
        label=str(fc)[0:2] + " [kHz]",
        linestyle=linestyles[i],
        linewidth=2.5
    )
    i -= 1
ax1.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1), fontsize='15')
# offset polar axes by -90 degrees
ax1.set_theta_offset(np.pi / 2)
# set theta direction to clockwise
ax1.set_theta_direction(-1)
# more theta ticks
ax1.set_xticks(np.linspace(0, 2 * np.pi, 18, endpoint=False))
# less radial ticks
ax1.set_yticks(np.linspace(-40, 0, 5))
ax1.set_rlabel_position(100)
ax1.set_rlabel_position(-90)

i = 3
for fc in central_freq[4:8]:
    rad_patt = np.mean(
        radiance[:, (freqs < fc + BW) & (freqs > fc - BW)], axis=1
    )
    rad_patt_norm = rad_patt / np.max(rad_patt)
    rad_patt_norm_dB = 20 * np.log10(rad_patt_norm)
    rad_patt_norm_dB = np.append(rad_patt_norm_dB, rad_patt_norm_dB[0])
    ax2.plot(
        np.deg2rad(theta),
        rad_patt_norm_dB,
        label=str(fc)[0:2] + " [kHz]",
        linestyle=linestyles[i],
        linewidth=2.5
    )
    i -= 1
ax2.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1), fontsize='15')
# offset polar axes by -90 degrees
ax2.set_theta_offset(np.pi / 2)
# set theta direction to clockwise
ax2.set_theta_direction(-1)
# more theta ticks
ax2.set_xticks(np.linspace(0, 2 * np.pi, 18, endpoint=False))
# less radial ticks
ax2.set_yticks(np.linspace(-40, 0, 5))
ax2.set_rlabel_position(100)
ax2.set_rlabel_position(-90)

plt.tight_layout()
plt.show()
# %%
fig.savefig('radiation', transparent=True)
# %% Mean radiance pattern display

rad_patt = np.mean(radiance, axis=1)
rad_patt_norm = rad_patt / np.max(rad_patt)
rad_patt_norm_dB = 20 * np.log10(rad_patt_norm)
rad_patt_norm_dB = np.append(rad_patt_norm_dB, rad_patt_norm_dB[0])

fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
ax.plot(np.deg2rad(theta), rad_patt_norm_dB)
# offset polar axes by -90 degrees
ax.set_theta_offset(np.pi / 2)
# set theta direction to clockwise
ax.set_theta_direction(-1)
# more theta ticks
ax.set_xticks(np.linspace(0, 2 * np.pi, 18, endpoint=False))
ax.set_ylabel("dB")
# less radial ticks
ax.set_yticks(np.linspace(-40, 0, 5))
ax.set_rlabel_position(-90)
ax.set_title(
    "Senscomp Series 7000 Transducer Mean Radiance Pattern 10[kHz] - 95[kHz]"
)
