import matplotlib.pyplot as plt
import sounddevice as sd
import numpy as np 
import scipy.signal as signal
import queue
from smbus2 import SMBus
import time
import os
from thymiodirect import Connection 
from thymiodirect import Thymio
import random
from broadcast_pcmd3180 import activate_mics

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def get_soundcard_iostream(device_list):
    for i, each in enumerate(device_list):
        dev_name = each['name']
        asio_in_name = 'MCHStreamer' in dev_name
        if asio_in_name:
            return (i, i)
        
def pow_two_pad_and_window(vec, show=False):
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

def sonar(signals, output_sig):
    obst_distance = 0
    for s in signals:
        filtered_signal = signal.correlate(s, output_sig, 'same', method='fft').transpose()
        smoothed_signal = np.abs(signal.hilbert(filtered_signal))
        peaks, _ = signal.find_peaks(smoothed_signal, prominence=7)
        if len(peaks) > 1:
            obst_distance += (peaks[1] - peaks[0])/fs*343/2 + 0.0325
        else:
            print('Skipped frame')
            return 0
    return obst_distance/signals.shape[0]

if __name__ == "__main__":

    fs = 192000
    dur = 3e-3

    t_tone = np.linspace(0, dur, int(fs*dur))
    chirp = signal.chirp(t_tone, 80e3, t_tone[-1], 20e3)    
    sig = pow_two_pad_and_window(chirp, show=False)

    silence_dur = 10 # [ms]
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

    device = get_soundcard_iostream(sd.query_devices())
    try:
        # real robot
        port = Connection.serial_default_port()
        th = Thymio(serial_port=port, 
                    on_connect=lambda node_id: print(f'Thymio {node_id} is connected'))
        th.connect()
        robot = th[th.first_node()]

        speed = 0
        rot_speed = 100
        # Delay to allow robot initialization of all variables
        time.sleep(1)
        current_time = time.time()
        activate_mics()
        while True:
            robot['motor.left.target'] = speed
            robot['motor.right.target'] = speed
            stream = sd.Stream(samplerate=fs,
                       blocksize=0, 
                       device=device, 
                       channels=(8, 1),
                       callback=callback,
                       latency=0.005)
            with stream:
                while stream.active:
                    pass
            current_frame = 0
            all_input_audio = []
            while not audio_in_data.empty():
                all_input_audio.append(audio_in_data.get())            
            input_audio = np.concatenate(all_input_audio)
            valid_channels_audio = input_audio.transpose()

            distance = sonar(valid_channels_audio, sig)*100

            print('Estimated distance: %3.1f' % distance, '[cm]')

            if distance < 20 and distance > 0:
                print('Encountered Obstacle')
                direction = random.choice(['l', 'r'])
                while(time.time() - current_time) < 1:
                    if direction == 'l':
                        robot['motor.left.target'] = -rot_speed
                        robot['motor.right.target'] = rot_speed
                    else:
                        robot['motor.left.target'] = rot_speed
                        robot['motor.right.target'] = -rot_speed
                current_time = time.time()
    
    except KeyboardInterrupt:
        robot['motor.left.target'] = 0
        robot['motor.right.target'] = 0
        print('Terminated by user')
    finally:
        try:
            th.disconnect()
        except Exception as e:
            print('Exception encountered:', e)
        finally:
            print('Fin')
