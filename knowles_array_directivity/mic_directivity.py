import soundfile as sf
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal, fft
import os

if __name__ == '__main__':
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    DIR = './recordings/'
    CUT_DIR = 'mic_'
    series = 2
    fs = 192000 # Hz
    durn = 3e-3
    t = np.linspace(0, durn, int(fs*durn))
    start_f, end_f = 15e3, 95e3
    sweep = signal.chirp(t, start_f, t[-1], end_f)
    sweep *= signal.windows.tukey(sweep.size, 0.2)
    sig = 0.8*sweep
    

    # audio_files = [f for f in os.listdir(DIR) if f.endswith('.wav')]
    # for i in np.arange(8):
    #     save_dir = os.path.join('series_' + str(series), CUT_DIR + str(i+1))
    #     print(save_dir)
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)       
    #     for f in audio_files:
    #         x = sf.read(os.path.join(DIR, f))[0]
    #         x = x[:, i]            
    #         xcorr = signal.correlate(x, sig, mode='same')
    #         xcorr_rolled = np.roll(xcorr, -len(sig)//2)
    #         envelope = np.abs(signal.hilbert(xcorr_rolled))
    #         idxs = signal.find_peaks(envelope, prominence=0.5, distance=int(30e-3*fs))[0]
    #         for n, idx in enumerate(idxs[5*(series-1):5*series]):
    #             x_trimmed = x[idx-384:idx + int(durn*fs) + 384]
    #             sf.write(save_dir + '/' + f[0:3] + 'deg_' + str(n+1) + '.wav', x_trimmed, int(fs))

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
            R = 1
            radiance = 4 * np.pi * R * np.abs(Channels_uni)
            radiances.append(radiance)

        radiances = np.array(radiances)
        mean_radiance = np.mean(radiances, axis=0)
        mean_radiances[k] = mean_radiance
        rad_patt = np.mean(radiance, axis=1)
        rad_patt_norm = rad_patt / np.max(rad_patt)
        rad_patt_norm_dB = 20 * np.log10(rad_patt_norm)
        rad_patt_norm_dB = np.append(rad_patt_norm_dB, rad_patt_norm_dB[0])

    
        ax.plot(np.deg2rad(theta), rad_patt_norm_dB, label='mic ' + str(k+1))
    # offset polar axes by -90 degrees
    ax.set_theta_offset(np.pi / 2)
    # set theta direction to clockwise
    ax.set_theta_direction(-1)
    # more theta ticks
    ax.set_xticks(np.linspace(0, 2 * np.pi, 18, endpoint=False))
    ax.set_ylabel("dB")
    # less radial ticks
    ax.set_yticks(np.linspace(-40, 0, 5))
    ax.set_rlabel_position(-90)
    ax.set_title('Microphones Directivity Patterns')
    ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))
    plt.tight_layout()
    plt.show()

    central_freq = np.array([20e3, 30e3, 40e3, 50e3, 60e3, 70e3, 80e3, 90e3])
    BW = 2e3

    linestyles = ["-", "--", "-.", ":"]

    for k in np.arange(8):
        fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "polar"})
        plt.suptitle("Microphone " + str(k+1) + " Directivity Pattern")
        i = 3
        for fc in central_freq[0:4]:
            rad_patt = np.mean(
                mean_radiances[k, :, (freqs < fc + BW) & (freqs > fc - BW)], axis=0
            )
            rad_patt_norm = rad_patt / np.max(rad_patt)
            rad_patt_norm_dB = 20 * np.log10(rad_patt_norm)
            rad_patt_norm_dB = np.append(rad_patt_norm_dB, rad_patt_norm_dB[0])
            ax1.plot(
                np.deg2rad(theta),
                rad_patt_norm_dB,
                label=str(fc)[0:2] + " [kHz]",
                linestyle=linestyles[i]
                )
            i -= 1
        ax1.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))
        # offset polar axes by -90 degrees
        ax1.set_theta_offset(np.pi / 2)
        # set theta direction to clockwise
        ax1.set_theta_direction(-1)
        # more theta ticks
        ax1.set_xticks(np.linspace(0, 2 * np.pi, 18, endpoint=False))
        # less radial ticks
        ax1.set_yticks(np.linspace(-40, 0, 5))
        ax1.set_rlabel_position(100)
        ax1.set_rlabel_position(-90)

        i = 3
        for fc in central_freq[4:8]:
            rad_patt = np.mean(
                mean_radiances[k, :, (freqs < fc + BW) & (freqs > fc - BW)], axis=0
            )
            rad_patt_norm = rad_patt / np.max(rad_patt)
            rad_patt_norm_dB = 20 * np.log10(rad_patt_norm)
            rad_patt_norm_dB = np.append(rad_patt_norm_dB, rad_patt_norm_dB[0])
            ax2.plot(
                np.deg2rad(theta),
                rad_patt_norm_dB,
                label=str(fc)[0:2] + " [kHz]",
                linestyle=linestyles[i]
            )
            i -= 1
        ax2.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))
        # offset polar axes by -90 degrees
        ax2.set_theta_offset(np.pi / 2)
        # set theta direction to clockwise
        ax2.set_theta_direction(-1)
        # more theta ticks
        ax2.set_xticks(np.linspace(0, 2 * np.pi, 18, endpoint=False))
        # less radial ticks
        ax2.set_yticks(np.linspace(-40, 0, 5))
        ax2.set_rlabel_position(100)
        ax2.set_rlabel_position(-90)

        plt.tight_layout()
        plt.show()
