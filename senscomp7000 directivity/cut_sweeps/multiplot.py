import soundfile
import numpy as np
import os
import matplotlib.pyplot as plt

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Load audio files, then plot them in a 6x6 grid
audio_files = os.listdir('./')  # List all files in the sweeps directory

# Create a 6x6 grid of subplots

fig, axs = plt.subplots(6, 6, figsize=(20, 20))

for i in range(6):  
    for j in range(6):
        # Load audio file
        audio, fs = soundfile.read('./' + audio_files[i*6+j])
        # Plot audio file
        axs[i, j].plot(np.linspace(0, len(audio)/fs, len(audio)), audio)
        axs[i, j].set_title(audio_files[i*6+j])
        axs[i, j].set_xlabel('Time (s)')
        axs[i, j].set_ylabel('Amplitude')
        # Shared x and y axes
        axs[i, j].sharex(axs[0, 0])
        axs[i, j].sharey(axs[0, 0])

plt.tight_layout()
plt.show()