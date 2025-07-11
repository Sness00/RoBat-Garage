import os
import soundfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
from scipy import fft, signal

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def pow_two_pad_and_window(vec):
    window = signal.windows.tukey(len(vec), alpha=0.2)
    padded_windowed_vec = vec * window
    # padded_windowed_vec = np.pad(windowed_vec, (0, 2**int(np.ceil(np.log2(len(windowed_vec)))) - len(windowed_vec)))
    return padded_windowed_vec/max(padded_windowed_vec)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"""
        \usepackage{lmodern}
        \renewcommand{\rmdefault}{cmr}
        \renewcommand{\sfdefault}{cmss}
        \renewcommand{\ttdefault}{cmtt}
    """
})

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
in_bands = True
# Load audio files, then plot them in a 6x6 grid
DIR = "./sanken_20250416/sweeps_1/"  # Directory containing the audio files
audio_files = os.listdir(DIR)  # List all files in the sweeps directory
NFFT = 2048
fs = 192e3
dur = 6e-3
hi_freq = 95e3
low_freq = 15e3
t_tone = np.linspace(0, dur, int(fs*dur))
chirp = signal.chirp(t_tone, hi_freq, t_tone[-1], low_freq)
sig = pow_two_pad_and_window(chirp)
S = fft.fft(sig, n=NFFT)[0:NFFT//2]
radiances = []
for i in np.arange(5):
    channels = []
    for j in np.arange(i, len(audio_files), 5):
        audio, fs = soundfile.read(DIR + audio_files[j])
        channels.append(audio)
    channels = np.array(channels)
    Channels = fft.fft(channels, n=NFFT, axis=1)
    Channels_uni = Channels[:, 0:NFFT//2]
    R = 1
    radiance = 4*np.pi*R*np.abs(Channels_uni)/np.abs(S)
    radiances.append(radiance)

radiances = np.array(radiances)
mean_radiance = np.mean(radiances, axis=0)
theta = np.linspace(0, 350, 36)
theta = np.append(theta, theta[0])

freqs = fft.fftfreq(NFFT, 1 / fs)
freqs = freqs[0:NFFT//2]


if in_bands:
    central_freq = np.array([20e3, 30e3, 40e3, 50e3, 60e3, 70e3, 80e3, 90e3])
    BW = 2e3
    i = 0
    k = 0
    for fc in central_freq:
        fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "polar"})
        rad_patt = np.mean(
            mean_radiance[:, (freqs < fc + BW) & (freqs > fc - BW)], axis=1
        )
        rad_patt_norm = rad_patt / np.max(rad_patt)
        rad_patt_norm_dB = 20 * np.log10(rad_patt_norm)
        rad_patt_norm_dB = np.append(rad_patt_norm_dB, rad_patt_norm_dB[0])
        ax.plot(
            np.deg2rad(theta),
            rad_patt_norm_dB,
            label=str(fc)[0:2] + " [kHz]",
            linewidth=1.5,
            color='k'
        )
    # ax1.legend(loc="upper right", bbox_to_anchor=(1.3, 1.2), fontsize=16)
    # offset polar axes by -90 degrees
        ax.set_theta_offset(np.pi / 2)
    # set theta direction to clockwise
        ax.set_theta_direction(-1)
    # more theta ticks
    # ax1.set_xticks(np.linspace(0, 2 * np.pi, 18, endpoint=False))
        ax.set_yticks(np.linspace(-40, 0, 5))
        ax.set_xticks(np.linspace(0, np.pi, 10))
        ax.tick_params(axis='y', labelsize=16)
        ax.tick_params(axis='x', labelsize=16)
        ax.set_ylabel("dB", fontdict={'fontsize': 16})
        ax.yaxis.label.set_rotation(0)
    # less radial ticks
    # ax1.set_rlabel_position(100)
    # ax1.set_rlabel_position(-90)
        ax.set_xlim(0, np.pi)
        ax.set_title(str(fc)[0:2] + " [kHz]", fontsize=16, weight='bold')
    # ax2.plot([])
    # ax2.plot([])
    # ax2.plot([])
    # ax2.plot([])
        # plt.savefig(str(fc)[0:2], dpi=1200, transparent=True)
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.show()
# %%
# fig.savefig('radiation', transparent=True)
# # %% Mean radiance pattern display

# rad_patt = np.mean(radiance, axis=1)
# rad_patt_norm = rad_patt / np.max(rad_patt)
# rad_patt_norm_dB = 20 * np.log10(rad_patt_norm)
# rad_patt_norm_dB = np.append(rad_patt_norm_dB, rad_patt_norm_dB[0])

# fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
# ax.plot(np.deg2rad(theta), rad_patt_norm_dB, linewidth=2, color='black')
# # offset polar axes by -90 degrees
# ax.set_theta_offset(np.pi / 2)
# # set theta direction to clockwise
# ax.set_theta_direction(-1)
# # more theta ticks
# ax.set_xticks(np.linspace(0, 2 * np.pi, 18, endpoint=False))
# for label in ax.get_xticklabels():
#     label.set_fontsize(16)  # Set to desired font size

# ax.set_ylabel("dB", {'fontsize': 16}, labelpad=20)
# ax.yaxis.label.set_rotation(0)
# ax.yaxis.label.set_ha('right')
# # less radial ticks
# ax.set_yticks(ticks=np.linspace(-40, 0, 5),
#                labels=['-40', '-30', '-20', '-10', '0'], fontsize=16)
# # for label in ax.get_yticklabels():
# #     label.set_fontsize(16)  # Set to desired font size
# ax.set_title(
#     "Senscomp Series 7000 Transducer Mean Radiance Pattern 15[kHz] - 95[kHz]",
#     {'fontsize': 20}
# )
# ax.set_xlim(0, np.pi)
# plt.tight_layout()
# plt.show()
