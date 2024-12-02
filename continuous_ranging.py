#%%
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import librosa
import numpy as np 
import scipy.signal as signal
import queue
from smbus2 import SMBus
import time as tm
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def start_mics():
    with SMBus(1) as bus:
        if bus.read_byte_data(int('4E', 16), int('75', 16)) != int('60', 16):
            bus.write_byte_data(int('4E', 16), int('2', 16), int('81', 16))
            tm.sleep(1e-3)
            bus.write_byte_data(int('4E', 16), int('7', 16), int('60', 16))
            bus.write_byte_data(int('4E', 16), int('B', 16), int('0', 16))
            bus.write_byte_data(int('4E', 16), int('C', 16), int('20', 16))
            bus.write_byte_data(int('4E', 16), int('22', 16), int('41', 16))
            bus.write_byte_data(int('4E', 16), int('2B', 16), int('40', 16))
            bus.write_byte_data(int('4E', 16), int('73', 16), int('C0', 16))
            bus.write_byte_data(int('4E', 16), int('74', 16), int('C0', 16))
            bus.write_byte_data(int('4E', 16), int('75', 16), int('60', 16))

def get_soundcard_iostream(device_list):
    for i, each in enumerate(device_list):
        dev_name = each['name']
        asio_in_name = 'MCHStreamer' in dev_name
        if asio_in_name:
            return (i, i)
        
def pow_two_pad_and_window(vec, show=False):
    padded_vec = np.pad(vec, (0, 2**int(np.ceil(np.log2(len(vec)))) - len(vec)))
    window = signal.windows.hann(len(padded_vec))
    padded_windowed_vec = padded_vec * window
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
    dur = 2.5e-3
    t_tone = np.linspace(0, dur, int(fs*dur))
    chirp = signal.chirp(t_tone, 80e3, t_tone[-1], 20e3)    
    sig = pow_two_pad_and_window(chirp, show=False)
    silence_dur = 100 # [ms]
    output_sig = np.float32(np.reshape(sig, (-1, 1)))

    audio_in_data = queue.Queue()
    
    # Stream callback function
    current_frame = 0
    def callback(indata, outdata, frames, time, status):
        global current_frame
        if status:
            print(status)
        chunksize = min(len(output_sig) - current_frame, frames)
        outdata[:chunksize] = output_sig[current_frame:current_frame + chunksize]
        current_frame += chunksize    
        if chunksize < frames:
            outdata[chunksize:] = 0
            current_frame = 0
        audio_in_data.put(indata.copy())
        
    stream = sd.Stream(samplerate=fs,
                       blocksize=0, 
                       device=get_soundcard_iostream(sd.query_devices()), 
                       channels=(8, 1),
                       callback=callback,
                       latency='high')
    tm.sleep(0.5)

    start_mics()
    try:
        N = 32
        w = signal.windows.hann(N)
        with stream:
            while True:
                sd.sleep(silence_dur)
                all_input_audio = []
                while not audio_in_data.empty():
                    all_input_audio.append(audio_in_data.get())            
                input_audio = np.concatenate(all_input_audio)
                valid_channels_audio = [input_audio[:, 2], input_audio[:, 3], input_audio[:, 6], input_audio[:, 7]]
                filtered_signals = np.zeros_like(valid_channels_audio)
                for i, rec in enumerate(valid_channels_audio):
                    filtered_signals[i, :] = np.abs(signal.correlate(rec, sig, 'same'))**2
                filt_padded_signals = np.pad(filtered_signals, ((0, 0), (N//2, N//2)))
                energy_local = np.zeros_like(filtered_signals)
                for i in np.arange(filtered_signals.shape[0]):
                    for j in np.arange(filtered_signals.shape[1]):
                        energy_local[i, j] = np.sum(filt_padded_signals[i, j : j + N] * w)
                obst_distance = 0
                peaks = []
                for en in energy_local:
                    peaks, _ = signal.find_peaks(en, prominence=25)
                    if len(peaks) > 1:
                        obst_distance += (peaks[1] - peaks[0])/fs*343.0/2 * 0.25
                    else:
                        continue
                print('%.4f' % obst_distance, '[m]')
                with audio_in_data.mutex:
                    audio_in_data.queue.clear()
    except KeyboardInterrupt:
        print('Keyboard Interrupt')