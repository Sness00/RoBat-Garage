# -*- coding: utf-8 -*-
"""
Example run-through
===================
This IS NOT a complete run-through. The follow things need to implemented by the user
    * playback signal segmentation 
    * bandpass filtering / other cleaning
    * matched-filtering to remove reflections

Other things to remember
------------------------
The GRAS microphone and target microphone may or may not be recorded with the same
analog-digital-converter!! Be Very Aware of this while comparing the sensitivities 
between the two microphones!!

Experimental notes
------------------
* Remember to write down all the GAIN values in dB for every recording you make.

WARNING
-------
As of APRIL 2025 -  this is a prototype module only. 
See below for the steps you need to take to ensure everything makes sense.


Always double-check with a 'validation' signal
----------------------------------------------
Test your calibration outcomes with a second signal that wasn't used anywhere before
and also recorded by both the GRAs and target mic. 

Created on Sat Apr 12 07:35:50 2025

@author: theja
"""
import os
import numpy as np 
import soundfile as sf
import matplotlib.pyplot as plt
os.chdir(os.path.abspath(os.path.dirname(__file__))) 
from utilities import *
from scipy import fft

# Load the substitution-calibration audio (calibration and target mic)
plt.rcParams['text.usetex'] = True
fs = sf.info('calibration/calibration_tone.wav').samplerate
print(fs)
gras_pbk_audio = sf.read('calibration/gras1/0_gras_recording.wav')[0]
# Load the 1 Pa reference tone 
gras_1Pa_tone = sf.read('calibration/calibration_tone.wav', start=int(fs*0.5),
                        stop=int(fs*1.5))[0]
# Gain compensate the audio (e.g. un-gain them all) to bring them to the 'same' 
# baseline level - not relevant for Ro-BAT
gras_pbk_gain = 30 # dB
gras_tone_gain = 30
gras_1Pa_tone *= db_to_linear(-gras_tone_gain)
gras_pbk_audio *= db_to_linear(-gras_pbk_gain)

# Calibration mic: Calculate the rms_Pascal of the 1 Pa calibration tone
rms_1Pa_tone = rms(gras_1Pa_tone)
print(f'The calibration mic has a sensitivity of {np.round(rms_1Pa_tone,3)}rms/Pa. RMS relevant only for this ADC!')
# dBSPL_to_rms = 94 - 20*np.log10(rms_1Pa_tone) # convert to dB SPL re 20 uPa
# GPA = fft.rfft(gras_pbk_audio) *2/ len(gras_pbk_audio) / np.sqrt(2) # get the FFT of the audio
# # Calculate the frequency bins
# gras_freqs = fft.rfftfreq(len(gras_pbk_audio), d=1/fs)
# # plot the frequency response of the GRAS mic
# plt.figure(figsize=(10, 5))
# plt.plot(gras_freqs, 20*np.log10(np.abs(GPA)) + dBSPL_to_rms, label='GRAS mic', color='blue')
# plt.xlim(10e3, 96e3)
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('SPL [dB] ref 20[$\\mu$Pa]')
# plt.title('Frequency response of GRAS mic')
# plt.grid()
# plt.tight_layout()
# plt.show()
# Now measure mic RMS over all frequency bands
gras_centrefreqs, gras_freqrms = calc_native_freqwise_rms(gras_pbk_audio, fs)
# Convert from RMS to Pascals (rms equivalent) since we know the GRAS sensitivity
gras_freqParms = gras_freqrms/rms_1Pa_tone # now the levels of each freq band in Pa_rms
# Also load the 'validation' calibration mic recording of the same sound
gras_rec, fs = sf.read('calibration/gras1/1_gras_recording.wav')
gras_rec *= db_to_linear(-gras_pbk_gain)
gras_centrefreqs, gras_freqrms = calc_native_freqwise_rms(gras_rec, fs)
gras_Pa = gras_freqrms/rms_1Pa_tone
gras_dbspl = pascal_to_dbspl(gras_Pa)

sennheiser_gain = 0
fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4))
for i in range(8):
# inside the loop
    sennheiser_pbk_audio = sf.read('calibration/mic_' + str(i+1) + '/0_20250526_19-46-54.wav')[0] 
    sennheiser_pbk_audio *= db_to_linear(-sennheiser_gain)
    sennheiser_centrefreqs, sennheiser_freqrms = calc_native_freqwise_rms(sennheiser_pbk_audio, fs)
    # Now let's calculate the RMS/Pa sensitivity using the knowledge from the 
    # calibration mic
    sennheiser_sensitivity = np.array(sennheiser_freqrms)/np.array(gras_freqParms)
    recorded_sound, fs = sf.read('calibration/mic_' + str(i+1) + '/1_20250526_19-46-54.wav')
    recorded_sound *= db_to_linear(-sennheiser_gain)
    # And finally let's check that the Sennheiser calibration makes sense
    # using a sound that we didn't use to calculate the sensitivity
    # If the length of the recorded target mic audio here is not the same as the calibration audio. 
    #  then you'll need to interpolate the microphone sensitivity using interpolate_freq_response in the
    # utilities.py module
    recsound_centrefreqs, freqwise_rms = calc_native_freqwise_rms(recorded_sound, fs)
    interp_sensitivity = interpolate_freq_response([sennheiser_centrefreqs, sennheiser_sensitivity],
                            recsound_centrefreqs)
    freqwise_Parms = freqwise_rms/interp_sensitivity # go from rms to Pa(rmseq.)
    freqwiese_dbspl = pascal_to_dbspl(freqwise_Parms)
    ax2.plot(sennheiser_centrefreqs, 20*np.log10(sennheiser_sensitivity), label=f'Mic {i+1}', linewidth=1.2)
ax2.set_xlabel('Frequency [Hz]', fontsize=16)
ax2.set_title('Knowles SPH0641LU4H-1 sensitivity', fontsize=20)
ax2.set_ylabel('Sensitivity [dB rms/Pa]', fontsize=16)
ax2.set_xlim(10e3, 96e3)
ax2.grid(True)
ax2.legend(fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.show()

# plt.figure()
# plt.plot(gras_centrefreqs,gras_dbspl, label='gras')
# plt.plot(recsound_centrefreqs,freqwiese_dbspl, label='sanken')
# plt.ylabel('dBrms SPL, re 20$\\mu$Pa', fontsize=12)
# plt.xlabel('Frequency, Hz', fontsize=12)
# plt.xlim(10e3, 96e3)
# plt.legend();

# plt.figure()
# plt.subplot(311)
# plt.title('Calibration tone')
# plt.plot(np.linspace(0, len(gras_1Pa_tone)/fs, len(gras_1Pa_tone)), gras_1Pa_tone)
# plt.subplot(312)
# plt.title('Calibration mic recording')
# plt.plot(np.linspace(0, len(gras_pbk_audio)/fs, len(gras_pbk_audio)), gras_pbk_audio)
# plt.subplot(313)
# plt.title('Target mic recording')
# plt.plot(np.linspace(0, len(sennheiser_pbk_audio)/fs, len(sennheiser_pbk_audio)), sennheiser_pbk_audio)
# plt.tight_layout()


# plt.figure()
# plt.subplot(211)
# plt.title('Calibration mic recording')
# plt.specgram(gras_pbk_audio, Fs=fs, NFFT=128, noverlap=64)
# plt.subplot(212)
# plt.title('Target mic recording')
# plt.specgram(sennheiser_pbk_audio, Fs=fs, NFFT=128, noverlap=64)
# plt.tight_layout() 
# #%%
# # Do any bandpass filtering/cleaning here
# raise NotImplementedError('Bandpass or audio cleaning steps not implemented')

# #%%
# # REMEMBER TO SEGMENT playback signal from calibration and target mic HERE -- THIS IS NOTT IMPLEMENTED!
# raise NotImplementedError('Remember to segment your sound here - you should have ideally nothing else but your playback sound')

# #%%
# # REMEMBER TO MATCH-FILTER (NOT IMPLEMENTED HERE) and extract only the direct path
# raise NotImplementedError('Matched filtering not implemented here - implement please')

# #%%
# # Check the SNR at the spectral level - use a silent audio clip from the above recordings
# # to be sure that your measurements mean something useful. Remember garbage in, garbage out.

# raise NotImplementedError('Check your SNR from the ambient sound!')
# # snr_target = dB(bandwise_tgtmic/tgt_silence_bandwise)
# # snr_gras = dB(bandwise_grasmic/gras_silence_bandwise)





# plt.figure()

# a0 = plt.subplot(211)
# plt.plot(gras_centrefreqs, gras_freqParms)
# plt.ylabel('Pressure_rmseqv., Pa', fontsize=12)
# plt.xlim(10e3, 96e3)
# plt.title('GRAS mic recording of playback')
# plt.subplot(212, sharex=a0)
# plt.plot(gras_centrefreqs, pascal_to_dbspl(gras_freqParms))
# plt.xlim(10e3, 96e3)
# plt.xlabel('Frequencies, Hz', fontsize=12)
# plt.ylabel('Sound pressure level,\n dBrms SPL re 20$\\mu $Pa', fontsize=12);


# Target microphone. Here we'll cover the case where we only get an RMS/Pa
# sensitivity. The other option is to calculate a mV/Pa sensitivity - which allows
# you to use the mic aross different ADCs - but also needs more info on the ADC specs


# plt.figure()

# a0 = plt.subplot(211)
# plt.plot(sennheiser_centrefreqs, sennheiser_freqrms)
# plt.ylabel('Pressure_rmseqv., Pa', fontsize=12)
# plt.title('Sanken mic recording of playback')
# plt.xlim(10e3, 96e3)
# plt.subplot(212, sharex=a0)
# plt.plot(sennheiser_centrefreqs, dB(sennheiser_freqrms))
# plt.xlabel('Frequencies, Hz', fontsize=12)
# plt.ylabel('dBrms a.u.', fontsize=12)
# plt.xlim(10e3, 96e3);

# plt.figure()
# a0 = plt.subplot(211)
# plt.plot(sennheiser_centrefreqs, sennheiser_sensitivity)
# plt.ylabel('a.u. RMS/Pa', fontsize=12)
# plt.title('Target mic sensitivity')
# plt.xlim(10e3, 96e3)
# # plt.ylim(0, 0.1)
# plt.subplot(212, sharex=a0)
# plt.plot(sennheiser_centrefreqs, dB(sennheiser_sensitivity))
# plt.xlabel('Frequencies, Hz', fontsize=12)
# plt.ylabel('dB a.u. rms/Pa', fontsize=12)
# # plt.ylim(-60,-10)
# plt.xlim(10e3, 96e3);


# We now have the target mic sensitivity - how do we use it to calculate the
# actual dB SPL? 

# Here we load a separate 'recorded sound' - a 'validation' audio clip let's call it 



# Now we know the sensitivity of the target mic - let's finally calculate
# the dB SPL of the recorded sound!
# We rely on combining the Pa rms of all relevant frequencies 
# e.g. see https://electronics.stackexchange.com/questions/642109/summing-rms-values-proof

# frequency_band = [15e3, 95e3] # min, max frequency to do the compensation Hz

# # Choose to calculate the dB SPL only for the frequency range of interest.
# # Target mic
# tgtmic_relevant_freqs = np.logical_and(recsound_centrefreqs>=frequency_band[0],
#                                 recsound_centrefreqs<=frequency_band[1])
# total_rms_freqwise_Parms = np.sqrt(np.sum(freqwise_Parms[tgtmic_relevant_freqs]**2))

# # Ground truth GRAS mic audio of the same sound. Here we use only the relevant 
# # frequency band of the recorded sweep
# gras_relevant_freqs = np.logical_and(gras_centrefreqs>=frequency_band[0],
#                                 gras_centrefreqs<=frequency_band[1])

# #%% This is one way to do it - use the RAW audio, and calculate its rms
# # since we the overall flat sensitivity of the GRAS
# gras_overallaudio_Parms = rms(gras_rec)/rms_1Pa_tone
# # This is the other way to do it by combining RMS values - like we did for the Sennheiser
# gras_totalrms_Parms = np.sqrt(np.sum(gras_Pa[gras_relevant_freqs]**2))

# print(f'GRAS dBrms SPL measures:{pascal_to_dbspl(gras_overallaudio_Parms)}, {pascal_to_dbspl(gras_totalrms_Parms)}')
# print(f'Sennheiser dBrms SPL measure: {pascal_to_dbspl(total_rms_freqwise_Parms)}')

