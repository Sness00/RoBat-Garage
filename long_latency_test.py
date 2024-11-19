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
from IPython.display import Audio
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

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
        
def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = fs/2
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data)
    return y

#%%
if __name__ == "__main__":
    x, fs = librosa.load('./test_audio.mp3', sr=192000)
    dur = len(x) / fs
    t = np.linspace(0, dur, len(x))

    fc = 45e3
    carrier = np.cos(2*np.pi*fc*t)

    mod_sig = x * carrier
    # mod_sig = x
    output_sig = np.float32(np.reshape(mod_sig, (-1, 1)))
    audio_in_data = queue.Queue()
    
    current_frame = 0
    def callback(indata, outdata, frames, time, status):
        audio_in_data.put(indata.copy())
        global current_frame
        if status:
            print(status)
        chunksize = min(len(output_sig) - current_frame, frames)
        # print('Time stamp in callback: ', tm.time())
        outdata[:chunksize] = output_sig[current_frame:current_frame + chunksize]
        # print('Time stamp in callback: ', tm.time())
        if chunksize < frames:
            outdata[chunksize:] = 0
            raise sd.CallbackStop()
        current_frame += chunksize

    start_mics()
    stream = sd.Stream(samplerate=fs,
                       blocksize=2**12, 
                       device=get_soundcard_iostream(sd.query_devices()), 
                       channels=(3, 1),
                       callback=callback)
    tm.sleep(0.5)
    t_start = tm.time()
    with stream:
        # print('Time stamp in stream context: ', tm.time())
        while (tm.time() - t_start) < 38:
           pass
    all_input_audio = []
    while not audio_in_data.empty():
        all_input_audio.append(audio_in_data.get())            
    input_audio = np.concatenate(all_input_audio)

#%%
    rec_audio = input_audio[0:len(carrier), 2]
    demod_audio = butter_lowpass_filter(rec_audio * carrier, 48000, fs, 4)

    cc = signal.correlate(demod_audio, x, 'full')
    cc_max_idx = np.argmax(np.abs(cc)) - len(carrier)
    lag = (cc_max_idx) / fs*1000
    print(lag)
    plt.figure()
    plt.plot(np.abs(cc))
    plt.show()

#%%
    plt.figure()
    aa = plt.subplot(211) 
    plt.plot(t, x)    
    plt.subplot(212, sharex=aa)
    plt.plot(t, demod_audio)
    plt.show()
#%%
    sf.write('demod_audio.wav', demod_audio, fs)