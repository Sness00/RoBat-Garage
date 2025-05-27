import soundfile as sf
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal, fft
import os

# def extract_noise_floor(files, fs, apply_symmetry=False):
#     cut_dir = 'noise_floor'
#     if not os.path.exists(rec_dir + '/' + cut_dir):
#         os.makedirs(rec_dir + '/' + cut_dir)
#     for file_name in files:
#         print(file_name)
#         x = sf.read(rec_dir + '/' + file_name, start=int(fs), frames=int(fs*0.01))[0]
#         sf.write(rec_dir + '/' + cut_dir + '/' + file_name, x, int(fs))
#         if apply_symmetry:
#             if int(file_name[0:3]) > 0 and int(file_name[0:3]) < 180:
#                     x_symm = x
#                     file_name_symm = str(360 - int(file_name[0:3])) + 'deg' + '.wav'
#                     sf.write(rec_dir + '/' + cut_dir + '/' + file_name_symm, x_symm, int(fs))

if __name__ == '__main__':
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    plt.rcParams['text.usetex'] = True
    PREPROCESS = False
    SNR = False
    IN_BANDS = True
    DIR = './recordings/'
    CUT_DIR = 'mic_'
    series = 1
    fs = 192000 # Hz
    durn = 3e-3
    t = np.linspace(0, durn, int(fs*durn))
    start_f, end_f = 15e3, 95e3
    sweep = signal.chirp(t, start_f, t[-1], end_f)
    sweep *= signal.windows.tukey(sweep.size, 0.2)
    sig = 0.8*sweep
    
    if PREPROCESS:
        # Extract signals
        audio_files = [f for f in os.listdir(DIR) if f.endswith('.wav')]
        for i in np.arange(8):
            save_dir = os.path.join('series_' + str(series), CUT_DIR + str(i+1))
            print(save_dir)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)       
            for f in audio_files:
                x = sf.read(os.path.join(DIR, f))[0]
                x = x[:, i]            
                xcorr = signal.correlate(x, sig, mode='same')
                xcorr_rolled = np.roll(xcorr, -len(sig)//2)
                envelope = np.abs(signal.hilbert(xcorr_rolled))
                idxs = signal.find_peaks(envelope, prominence=0.5, distance=int(30e-3*fs))[0]
                for n, idx in enumerate(idxs[5*(series-1):5*series]):
                    x_trimmed = x[idx-384:idx + int(durn*fs) + 384]
                    sf.write(save_dir + '/' + f[0:3] + 'deg_' + str(n+1) + '.wav', x_trimmed, int(fs))

        # Extract noise floor
        audio_files = [f for f in os.listdir(DIR) if f.endswith('.wav')]
        for i in np.arange(8):
            save_dir = os.path.join('noise_floor', CUT_DIR + str(i+1))
            print(save_dir)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for f in audio_files:
                x = sf.read(os.path.join(DIR, f), start=int(0.01*fs), frames=int(fs*0.01))[0]
                x = x[:, i]
                sf.write(save_dir + '/' + f[0:3] + 'deg' + '.wav', x, int(fs))
    
    if SNR:
        fig, ax = plt.subplots(8, 1)
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        plt.suptitle('SNR - Knowles SPH0641LU4H-1 array') 
        SIG_DIR = './series_' + str(series) + '/' + CUT_DIR  # Directory containing the audio files        
        NOISE_DIR = './noise_floor/' + CUT_DIR  # Directory containing the audio files
       
       
        for k in np.arange(8):
            snrs = []
            signal_files = [f for f in os.listdir(SIG_DIR + str(k+1)) if f.endswith('.wav')]
            noise_files = os.listdir(NOISE_DIR + str(k+1))
            for i in np.arange(36):
                noise = sf.read(NOISE_DIR + str(k+1) + './' + noise_files[i])[0]
                snr = 0
                for j in np.arange(5*i, 5*(i + 1)):
                    sig = sf.read(SIG_DIR + str(k+1) + './' + signal_files[j])[0]
                    # Compute the SNR
                    snr += 10 * np.log10(np.mean(sig**2) / np.mean(noise**2))
                snrs.append(snr / 5)
            print(np.array(snrs).shape)
            
            ax[k].stem(np.linspace(0, 350, 36), snrs, markerfmt="o", basefmt=" ")
            # plt.title("SNR - Senscomp Series 7000 Transducer")
            ax[k].set_xlabel("Angle [deg]")
            ax[k].set_ylabel("SNR [dB]")
            ax[k].grid()
            ax[k].set_title("Microphone " + str(k+1))
        plt.tight_layout()
        plt.show()
        


    NFFT = 2048
    theta = np.linspace(0, 350, 36)
    theta = np.append(theta, theta[0])
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    mean_radiances = np.zeros((8, 36, NFFT//2))
    for k in np.arange(8):
        radiances = []
        audio_files = [f for f in os.listdir(os.path.join('series_' + str(series), CUT_DIR + str(k+1))) if f.endswith('.wav')]
        for i in np.arange(5):
            channels = []
            for j in np.arange(i, len(audio_files), 5):
                audio, fs = sf.read('series_' + str(series) + '/' + CUT_DIR + str(k+1) + '/' + audio_files[j])
                channels.append(audio)
            channels = np.array(channels)

            Channels = fft.fft(channels, n=NFFT, axis=1)
            Channels_uni = Channels[:, 0:NFFT//2]
            freqs = fft.fftfreq(NFFT, 1 / fs)
            freqs = freqs[0:NFFT//2]
            radiance = (np.abs(Channels)**2/NFFT)[:, 0:NFFT//2]
            radiances.append(radiance)

        radiances = np.array(radiances)
        mean_radiance = np.mean(radiances, axis=0)
        mean_radiances[k] = mean_radiance
        rad_patt = np.mean(radiance, axis=1)
        rad_patt_norm = rad_patt / np.max(rad_patt)
        rad_patt_norm_dB = 20 * np.log10(rad_patt_norm)
        rad_patt_norm_dB = np.append(rad_patt_norm_dB, rad_patt_norm_dB[0])

    
        ax.plot(np.deg2rad(theta), rad_patt_norm_dB, label='mic ' + str(k+1), linestyle=('--' if k==5 else '-'))
    # offset polar axes by -90 degrees
    ax.set_theta_offset(np.pi / 2)
    # set theta direction to clockwise
    ax.set_theta_direction(-1)
    # more theta ticks
    ax.set_xticks(np.linspace(0, 2 * np.pi, 18, endpoint=False))
    ax.set_ylabel("dB", labelpad=20, fontsize=16, y=0.7)
    # less radial ticks
    # ax.set_yticks(np.arange(-60, 0, 2), labels=[str(i) for i in np.arange(-60, 0, 2)], fontsize=5)
    ax.set_rlabel_position(-70)
    ax.set_ylim(-70, 1)
    ax.yaxis.label.set_rotation(0)
    ax.set_title('Knowles SPH0641LU4H-1 directivity', fontsize=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1), fontsize=16)
    for label in ax.get_yticklabels():
        label.set_fontsize(16)
    for label in ax.get_xticklabels():
        label.set_fontsize(16)
    plt.tight_layout()
    # plt.show()

    if IN_BANDS:
        central_freq = np.array([20e3, 30e3, 40e3, 50e3, 60e3, 70e3, 80e3, 90e3])
        BW = 2e3
        fig, ax1 = plt.subplots(4, 2, subplot_kw={"projection": "polar"})
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        plt.suptitle("Knowles SPH0641LU4H-1 directivity", fontsize=20)

        for k in np.arange(len(central_freq)):
            for j in np.arange(8):
                rad_patt = np.mean(
                    mean_radiances[j, :, (freqs < central_freq[k] + BW) & (freqs > central_freq[k] - BW)], axis=0
                )
                rad_patt_norm = rad_patt / np.max(rad_patt)
                rad_patt_norm_dB = 20 * np.log10(rad_patt_norm)
                rad_patt_norm_dB = np.append(rad_patt_norm_dB, rad_patt_norm_dB[0])
                ax1[k%4, k//4].plot(
                    np.deg2rad(theta),
                    rad_patt_norm_dB,
                    label='Mic ' + str(j+1),
                    linestyle=('--' if j==5 else '-')
                )
                ax1[k%4, k//4].set_title(str(central_freq[k])[0:2] + " [kHz]")
                ax1[k%4, k//4].legend(loc="upper right", bbox_to_anchor=(1.3, 1.2))
                # offset polar axes by -90 degrees
                ax1[k%4, k//4].set_theta_offset(np.pi / 2)
                # set theta direction to clockwise
                ax1[k%4, k//4].set_theta_direction(-1)
                # more theta ticks
                ax1[k%4, k//4].set_xticks(np.linspace(0, 2 * np.pi, 18, endpoint=False))
                # less radial ticks
                ax1[k%4, k//4].set_yticks(np.linspace(-60, 0, 5))
                ax1[k%4, k//4].set_rlabel_position(-90)

    plt.tight_layout()
    plt.show()
