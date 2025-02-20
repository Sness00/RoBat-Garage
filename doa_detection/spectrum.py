import os
import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np 
import scipy.signal as signal
import queue
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from capon import capon_method

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
    dur = 2e-3
    hi_freq = 50e3
    low_freq = 30e3

    t_tone = np.linspace(0, dur, int(fs*dur))
    chirp = signal.chirp(t_tone, hi_freq, t_tone[-1], low_freq)    
    sig = pow_two_pad_and_window(chirp, fs, show=False)

    silence_dur = 15 # [ms]
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

    with stream:
        while stream.active:
            pass

    # Transfer input data from queue to an array
    all_input_audio = []
    while not audio_in_data.empty():
        all_input_audio.append(audio_in_data.get())            
    input_audio = np.concatenate(all_input_audio)
    db_rms = 20*np.log10(np.std(input_audio))
    if db_rms < -40:
        print('Low output level. Replace amp battery')
    else:
        valid_channels_audio = input_audio
        filtered_signals = signal.correlate(valid_channels_audio, np.reshape(sig, (-1, 1)), 'full', method='fft')
        envelopes = np.abs(signal.hilbert(filtered_signals, axis=0))

        peaks = []
        for e in envelopes.T:
            p, _ = signal.find_peaks(e, prominence=12)
            peaks.append(p[0])

        furthest_peak = np.max(peaks)

        theta2, p_capon = capon_method(filtered_signals[furthest_peak+70:furthest_peak+70+384, ], fs=fs, nch=filtered_signals.shape[1], d=0.003, bw=(low_freq, hi_freq))
        
        plt.figure()
        plt.plot(theta2, p_capon)
        plt.grid()
        plt.title('Fast Implementation')
        plt.tight_layout()
        plt.show()
