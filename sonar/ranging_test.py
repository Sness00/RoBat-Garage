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
        
def pow_two_pad_and_window(vec):
    vec = vec[0:479]
    n_samples = len(vec)
    i = 1
    while 2**i < n_samples:
        i += 1
    padded_vec = np.append(vec, np.zeros((2**i - n_samples, )))
    window = signal.windows.hann(2**i)
    padded_windowed_vec = padded_vec * window
    return padded_windowed_vec/max(padded_windowed_vec)

#%%
if __name__ == "__main__":

    # Load and resample at 192kHz the test audio
    x, fs = librosa.load('./1-80k_3ms.wav', sr=192000)
    sig = pow_two_pad_and_window(x)
    
    dur = len(sig) / fs

    # t = np.linspace(0, dur, len(sig))
    # plt.figure()
    # plt.plot(t, sig)
    # plt.show()
#%%
    recording_time = 0.05
    recording_samples = int(recording_time * fs)

    output_sig = np.float32(np.reshape(np.append(sig, np.zeros(recording_samples - len(sig), )), (-1, 1)))
    chunk = 0
    audio_in_data = queue.Queue()
    
    # Stream callback function
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
            raise sd.CallbackStop()
        current_frame += chunksize
        

    # Initialize and power on mics array
    start_mics()

    # Create stream
    stream = sd.Stream(samplerate=fs,
                       blocksize=chunk, 
                       device=get_soundcard_iostream(sd.query_devices()), 
                       channels=(8, 1),
                       callback=callback,
                       latency='low')
    
    # Little pause to let the soundcard settle
    tm.sleep(0.5)

    with stream:
        while stream.active:
            pass
    
    # Transfer input data from queue to an array
    all_input_audio = []
    while not audio_in_data.empty():
        all_input_audio.append(audio_in_data.get())            
    input_audio = np.concatenate(all_input_audio)[0:recording_samples]

    valid_channels_audio = [input_audio[:, 2], input_audio[:, 3], input_audio[:, 6], input_audio[:, 7]]

    filtered_signals = np.zeros_like(valid_channels_audio)

    for i, rec in enumerate(valid_channels_audio):
        filtered_signals[i, :] = np.abs(signal.correlate(rec, sig, 'same'))**2
        # filtered_signals[i, :] /= np.max(filtered_signals[i, :])

    N = 32
    w = signal.windows.hann(N)
    filt_padded_signals = np.pad(filtered_signals, ((0, 0), (N//2, N//2)))
    energy_local = np.zeros_like(filtered_signals)

    for i in np.arange(filtered_signals.shape[0]):
        for j in np.arange(filtered_signals.shape[1]):
            energy_local[i, j] = np.sum(filt_padded_signals[i, j : j + N] * w)

    peaks = []
    min_length = 1000
    for en in energy_local:
        idxs, _ = signal.find_peaks(en, prominence=14)
        if len(idxs) < min_length:
            min_length = len(idxs)
    
    for en in energy_local:
        idxs, _ = signal.find_peaks(en, prominence=14)
        peaks.append(idxs[0:min_length])
    peaks_array = np.array(peaks)

    if min_length > 1:       
        estimated_distances = []
        for i, p in enumerate(peaks):
            estimated_distances.append((p[1] - p[0])/fs*343.0/2)
            print('Estimated distance for channel', i, ':', '%.5f' % estimated_distances[i], '[m]')
    else:
        print('Only', min_length, 'peaks found')
    t_plot = np.linspace(0, recording_time, int(fs*recording_time))
    plt.figure()
    aa = plt.subplot(221)
    plt.plot(energy_local[0, :])
    plt.vlines(peaks_array[0, :], 0, 100, linestyles='dashed', colors='r')
    plt.title('Matched Filter Channel 3')
    plt.grid()
    plt.subplot(222, sharex=aa, sharey=aa)
    plt.plot(energy_local[1, :])
    plt.vlines(peaks_array[1, :], 0, 100, linestyles='dashed', colors='r')
    plt.title('Matched Filter Channel 4')
    plt.grid()
    plt.subplot(223, sharex=aa, sharey=aa)
    plt.plot(energy_local[2, :])
    plt.vlines(peaks_array[2, :], 0, 100, linestyles='dashed', colors='r')
    plt.title('Matched Filter Channel 7')
    plt.grid()
    plt.subplot(224, sharex=aa, sharey=aa)
    plt.plot(energy_local[3, :])
    plt.vlines(peaks_array[3, :], 0, 100, linestyles='dashed', colors='r')
    plt.title('Matched Filter Channel 8')
    plt.grid()
    plt.tight_layout()
    plt.show()

    # plt.figure()
    # aa = plt.subplot(211)
    # plt.plot(t_plot, valid_channels_audio[0])
    # # plt.xticks([0.005*i for i in range(20)], [str(5*i) for i in range(20)])
    # plt.xlabel('[ms]')
    # plt.title('Recorded Signal')
    # plt.grid()
    # plt.subplot(212, sharex=aa)
    # plt.plot(t_plot, filtered_signals[0, :])
    # plt.xlabel('[ms]')
    # plt.title('Matched Filter Output')
    # plt.grid()
    # plt.tight_layout()
    # plt.show()  
    
    # plt.figure()
    # plt.subplot(2, 2, 1)
    # plt.plot(t_plot, input_audio[:, 2])
    # plt.title('Channel 2')
    # plt.subplot(2, 2, 2)
    # plt.plot(t_plot, input_audio[:, 3])
    # plt.title('Channel 3')
    # plt.subplot(2, 2, 3)
    # plt.plot(t_plot, input_audio[:, 6])
    # plt.title('Channel 6')
    # plt.subplot(2, 2, 4)
    # plt.plot(t_plot, input_audio[:, 7])
    # plt.title('Channel 7')
    # plt.show()
      
    # plt.figure()
    # plt.subplot(2, 2, 1)
    # plt.plot(t_plot, filtered_signals[0, :])
    # plt.title('Channel 2')
    # plt.subplot(2, 2, 2)
    # plt.plot(t_plot, filtered_signals[1, :])
    # plt.title('Channel 3')
    # plt.subplot(2, 2, 3)
    # plt.plot(t_plot, filtered_signals[2, :])
    # plt.title('Channel 6')
    # plt.subplot(2, 2, 4)
    # plt.plot(t_plot, filtered_signals[3, :])
    # plt.title('Channel 7')
    # plt.show()

    # plt.figure()
    # aa = plt.subplot(221)
    # plt.plot(t_plot, valid_channels_audio[0])
    # plt.subplot(222, sharex=aa)
    # plt.plot(t_plot, filtered_signals[0, :])
    # plt.subplot(223)
    # plt.specgram(valid_channels_audio[0], Fs=fs, NFFT=1024, noverlap=512)
    # plt.subplot(224)
    # plt.specgram(filtered_signals[0, :], Fs=fs, NFFT=1024, noverlap=512)
    # plt.tight_layout()
    # plt.show()

    # plt.figure()
    # plt.suptitle('Ceiling Reflection Test - Ground Level')
    # aa = plt.subplot(211)
    # plt.specgram(valid_channels_audio[0], Fs=fs, NFFT=128, noverlap=64)
    # plt.title('Recorded Signal')
    # plt.subplot(212, sharex=aa)
    # plt.specgram(filtered_signals[0, :], Fs=fs, NFFT=128, noverlap=64)
    # plt.title('Matched Filter Output')
    # plt.grid()
    # plt.tight_layout()
    # plt.show()