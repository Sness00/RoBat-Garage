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
    c = 343
    d_max = 2.4e-2
    t_tone = np.linspace(0, dur, int(fs*dur))
    chirp = signal.chirp(t_tone, 80e3, t_tone[-1], 20e3)    
    sig = pow_two_pad_and_window(chirp, fs, show=False)

    silence_dur = 25 # [ms]
    silence_samples = int(silence_dur * fs/1000)
    silence_vec = np.zeros((silence_samples, ))
    full_sig = pow_two(np.concatenate((sig, silence_vec)))
    output_sig = np.float32(np.reshape(full_sig, (-1, 1)))

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

    soundcard = get_soundcard_iostream(sd.query_devices())

    stream = sd.Stream(samplerate=fs,
                       blocksize=0, 
                       device=soundcard, 
                       channels=(8, 1),
                       callback=callback,
                       latency='low')
    
    # Little pause to let the soundcard settle
    time.sleep(0.5)

    activate_mics()
    with stream:
        while stream.active:
            pass
    
    # Transfer input data from queue to an array
    all_input_audio = []
    while not audio_in_data.empty():
        all_input_audio.append(audio_in_data.get())            
    input_audio = np.concatenate(all_input_audio)

    valid_channels_audio = np.array([input_audio[:, 0], input_audio[:, 7]]).transpose()
    filtered_signals = signal.correlate(valid_channels_audio, np.reshape(sig, (-1, 1)), 'full', method='fft')
    envelopes = np.abs(signal.hilbert(filtered_signals, axis=0))

    # plt.figure()
    # plt.plot(envelopes)
    # plt.show()

    peaks = []
    enough = True
    for i in np.arange(2):
        # idxs, _ = signal.find_peaks(envelopes[:, i], prominence=8, distance=int(len(sig)/64))
        idxs, _ = signal.find_peaks(envelopes[:, i], prominence=6)
        if len(idxs) < 2:
            enough = False
        peaks.append(idxs[0:2])

    if not enough:
        print('Not enough peaks found')

    else:
        estimated_distances = []
        for i, p in enumerate(peaks):
            estimated_distances.append((p[1] - p[0])/fs*c/2 + 0.0325) # biased
            print('Estimated distance for channel', i, ':', '%.5f' % estimated_distances[i], '[m]')

    peaks_array = np.array(peaks)

    t_plot = np.linspace(0, input_audio.shape[0]*fs, input_audio.shape[0])
    plt.figure()
    aa = plt.subplot(211)
    plt.plot(envelopes[:, 0])
    plt.vlines(peaks_array[0, :], 0, 100, linestyles='dashed', colors='r')
    plt.title('Matched Filter Channel 1')
    plt.grid()
    plt.subplot(212, sharex=aa, sharey=aa)
    plt.plot(envelopes[:, 1])
    plt.vlines(peaks_array[1, :], 0, 100, linestyles='dashed', colors='r')
    plt.title('Matched Filter Channel 8')
    plt.grid()
    plt.tight_layout()
    plt.show()

    tau = ((peaks_array[1, 1] - peaks_array[1, 0]) - (peaks_array[0, 1] - peaks_array[0, 0]))/fs

    theta = 180/np.pi*np.arcsin(tau*c/d_max)
    print('Estimated angle: %3.1f' % theta, '[deg]')