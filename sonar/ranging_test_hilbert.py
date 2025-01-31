import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np 
import scipy.signal as signal
import queue
from smbus2 import SMBus
import time
import os
from broadcast_pcmd3180 import activate_mics

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def get_soundcard_iostream(device_list):
    for i, each in enumerate(device_list):
        dev_name = each['name']
        asio_in_name = 'MCHStreamer' in dev_name
        if asio_in_name:
            return (i, i)
        
def pow_two_pad_and_window(vec, fs, show=False):
    window = signal.windows.hann(len(vec))
    windowed_vec = vec * window
    padded_windowed_vec = np.pad(windowed_vec, (0, 2**int(np.ceil(np.log2(len(windowed_vec)))) - len(windowed_vec)))
    if show:
        dur = len(padded_windowed_vec) / fs
        t = np.linspace(0, dur, len(padded_windowed_vec))
        plt.figure()
        plt.plot(t, padded_windowed_vec)
        plt.show()
    return padded_windowed_vec/max(padded_windowed_vec)

def pow_two(vec):
    return np.pad(vec, (0, 2**int(np.ceil(np.log2(len(vec)))) - len(vec)))

if __name__ == "__main__":

    fs = 192000
    dur = 3e-3

    t_tone = np.linspace(0, dur, int(fs*dur))
    chirp = signal.chirp(t_tone, 80e3, t_tone[-1], 20e3)    
    sig = pow_two_pad_and_window(chirp, fs, show=False)

    silence_dur = 25 # [ms]
    silence_samples = int(silence_dur * fs/1000)
    silence_vec = np.zeros((silence_samples, ))
    full_sig = pow_two(np.concatenate((sig, silence_vec)))
    stereo_sig = np.hstack([full_sig.reshape(-1, 1), full_sig.reshape(-1, 1)])

    output_sig = np.float32(stereo_sig)

    audio_in_data = queue.Queue()

    current_frame = 0
    def callback(indata, outdata, frames, time, status):
        global current_frame
        if status:
            print(status)
        chunksize = min(len(output_sig) - current_frame, frames)
        outdata[:chunksize] = output_sig[current_frame:current_frame + chunksize]
        audio_in_data.put(indata.copy())
        if chunksize < frames:
            outdata[chunksize:] = 0
            raise sd.CallbackAbort()
        current_frame += chunksize

    activate_mics()
    soundcard = get_soundcard_iostream(sd.query_devices())

    stream = sd.Stream(samplerate=fs,
                       blocksize=0, 
                       device=soundcard, 
                       channels=(8, 2),
                       callback=callback,
                       latency=0.005)
    
    # Little pause to let the soundcard settle
    time.sleep(0.5)

    with stream:
        while stream.active:
            pass
    
    # Transfer input data from queue to an array
    all_input_audio = []
    while not audio_in_data.empty():
        all_input_audio.append(audio_in_data.get())            
    input_audio = np.concatenate(all_input_audio)

    valid_channels_audio = input_audio
    filtered_signals = signal.correlate(valid_channels_audio, np.reshape(sig, (-1, 1)), 'full', method='fft')
    envelopes = np.abs(signal.hilbert(filtered_signals, axis=0))

    peaks = []
    enough = True
    for i in np.arange(envelopes.shape[1]):
        # idxs, _ = signal.find_peaks(envelopes[:, i], prominence=8, distance=int(len(sig)/64))
        idxs, _ = signal.find_peaks(envelopes[:, i], prominence=7)
        if len(idxs) < 2:
            enough = False
        peaks.append(idxs[0:2])

    if not enough:
        print('Not enough peaks found')

    else:
        estimated_distances = []
        for i, p in enumerate(peaks):
            estimated_distances.append((p[1] - p[0])/fs*343.0/2 + 0.0325)
            print('Estimated distance for channel', i+1, ':', '%.5f' % estimated_distances[i], '[m]')    
        peaks_array = np.array(peaks)
 
    plt.figure()
    aa = plt.subplot(421)
    if enough:
        plt.vlines(peaks_array[0, :], 0, 100, linestyles='dashed', colors='r')
    plt.plot(envelopes[:, 0])
    plt.title('Envelope of Channel 1')
    plt.grid()

    for i in np.arange(1, envelopes.shape[1]):
        plt.subplot(4, 2, i+1, sharex=aa, sharey=aa)
        if enough:
            plt.vlines(peaks_array[i, :], 0, 100, linestyles='dashed', colors='r')
        plt.plot(envelopes[:, i])
        plt.title('Envelope of Channel %d' %(i+1))
        plt.grid()

    plt.tight_layout()
    plt.show()
    
    t_plot = np.linspace(0, input_audio.shape[0]/fs, input_audio.shape[0])
    plt.figure()
    for i in np.arange(0, input_audio.shape[1]):
        plt.subplot(4, 2, i+1)
        plt.plot(t_plot, input_audio[:, i])
        plt.title('Channel %d Audio' % (i+1))
    plt.tight_layout()
    plt.show()

    t_plot = np.linspace(0, filtered_signals.shape[0]/fs, filtered_signals.shape[0])
    plt.figure()
    for i in np.arange(0, filtered_signals.shape[1]):
        plt.subplot(4, 2, i+1)
        plt.plot(t_plot, filtered_signals[:, i])
        plt.title('Matched Filter Channel %d' % (i+1))
    plt.tight_layout()
    plt.show()
        