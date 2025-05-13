import ffmpeg
import numpy as np
import librosa
import io
import soundfile as sf
from matplotlib import pyplot as plt
import sys
import os
from scipy import signal

os.chdir(os.path.dirname(os.path.abspath(__file__)))
# Define path to video
input_path = './videos/GX010517.mp4'
robot_path = './audio/20250513_18-46-35.wav'
try:
    robot_recording, fs = sf.read(robot_path)
    # Run ffmpeg to extract audio and pipe as WAV
    out, _ = (
        ffmpeg
        .input(input_path)
        .output('pipe:', format='wav', acodec='pcm_s16le')
        .run(capture_stdout=True, capture_stderr=True)
    )

    # Load audio from bytes using soundfile
    audio_data, sr = librosa.load(io.BytesIO(out), sr=fs, mono=True)
    print(sr)
    # Convert to mono if stereo
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)   

    xcorr = np.roll(signal.correlate(audio_data, robot_recording[:, 0], mode='same'), -len(robot_recording) // 2)
    index = np.argmax(np.abs(xcorr))

    robot_audio = robot_recording[:, 0]

    t = np.linspace(0, len(xcorr) / sr, num=len(xcorr))

    plt.figure()
    plt.plot(robot_audio/np.max(np.abs(robot_audio)))
    plt.plot(np.roll(audio_data/np.max(np.abs(audio_data)), -index))    
    plt.plot(np.roll(xcorr/np.max(np.abs(xcorr)), -index))        
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()
except ffmpeg.Error as e:
    print('ffmpeg error:', e.stderr.decode(), file=sys.stderr)
    sys.exit(1)