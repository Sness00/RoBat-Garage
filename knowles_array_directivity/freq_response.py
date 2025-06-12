import os
import numpy as np 
import soundfile as sf
import matplotlib.pyplot as plt
os.chdir(os.path.abspath(os.path.dirname(__file__))) 
from utilities import *
from scipy import fft

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

# Load the substitution-calibration audio (calibration and target mic)
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
fs = sf.info('calibration/calibration_tone.wav').samplerate
# Load the 1 Pa reference tone 
gras_1Pa_tone = sf.read('calibration/calibration_tone.wav', start=int(fs*0.5),
                        stop=int(fs*1.5))[0]
# Gain compensate the audio (e.g. un-gain them all) to bring them to the 'same' 
# baseline level - not relevant for Ro-BAT
gras_pbk_gain = 0 # dB
gras_tone_gain = 0
gras_1Pa_tone *= db_to_linear(-gras_tone_gain)
# Calibration mic: Calculate the rms_Pascal of the 1 Pa calibration tone
rms_1Pa_tone = rms(gras_1Pa_tone)
# Convert from RMS to Pascals (rms equivalent) since we know the GRAS sensitivity
sennheiser_gain = 0

audio_files = os.listdir('calibration/gras1')
gras_freq_responses = []
for j in range(len(audio_files)):
    gras_pbk_audio = sf.read('calibration/gras1'+ '/' + audio_files[j])[0]
    gras_pbk_audio *= db_to_linear(-gras_pbk_gain)
    gras_centrefreqs, gras_freqrms = correct_rms(gras_pbk_audio, fs)
    gras_freq_responses.append(gras_freqrms)
gras_freqrms = np.mean(np.array(gras_freq_responses), axis=0)
gras_freqrms = moving_average(gras_freqrms, 32)
gras_freqParms = gras_freqrms/rms_1Pa_tone # now the levels of each freq band in Pa_rms

Pa_to_dBFS = -26
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 4))
for i in range(8):
# inside the loop
    audio_files = os.listdir('calibration/mic_' + str(i+1))
    freq_responses = []
    for j in range(len(audio_files)):
        sennheiser_pbk_audio = sf.read('calibration/mic_' + str(i+1) + '/' + audio_files[j])[0]
        sennheiser_pbk_audio *= db_to_linear(-sennheiser_gain)
        sennheiser_centrefreqs, sennheiser_freqrms = correct_rms(sennheiser_pbk_audio, fs)
        freq_responses.append(sennheiser_freqrms)
    sennheiser_freqrms = np.mean(np.array(freq_responses), axis=0)
    sennheiser_freqrms = moving_average(sennheiser_freqrms, 32)
    sennheiser_sensitivity = np.array(sennheiser_freqrms)/np.array(gras_freqParms)
    mic_sensitivity = sennheiser_sensitivity / 10**(Pa_to_dBFS/20)
    ax2.plot(sennheiser_centrefreqs, 20*np.log10(mic_sensitivity), label=f'Mic {i+1}', linewidth=1.2)
ax2.set_xlabel('Frequency [kHz]', fontsize=16)
ax2.set_title('Knowles SPH0641LU4H-1 sensitivity\nf$_C$$_K$=12.288 [MHz], V$_D$$_D$=3.3[V]', fontsize=20)
ax2.set_ylabel('Sensitivity [dBFS]', fontsize=16)
ax2.set_xlim(15e3, 96e3)
ax2.set_xticks(ticks=[20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000],
           labels=['20', '30', '40', '50', '60', '70', '80', '90'], fontsize=16)
ax2.grid(True)
ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 1])  # Adjust the right margin to make space
plt.savefig('mfr', dpi=1200, transparent=True)
# plt.show()
