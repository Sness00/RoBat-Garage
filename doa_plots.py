import os
import yaml
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from scipy import signal

if __name__ == "__main__":
    plt.rcParams.update({
"text.usetex": True,
"font.family": "serif",
"font.serif": ["Computer Modern Roman"],
"text.latex.preamble": r"""
\usepackage{lmodern}
\renewcommand{\rmdefault}{cmr}
\renewcommand{\sfdefault}{cmss}
\renewcommand{\ttdefault}{cmtt}
""",
    "font.size": 16,           # Set default font size
    "axes.labelsize": 16,      # Axis label font size
    "xtick.labelsize": 16,     # X tick label font size
    "ytick.labelsize": 16,     # Y tick label font size
    "legend.fontsize": 16,     # Legend font size
    "axes.titlesize": 16       # Title font size
})
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    # Set the path to the directory containing the .npy files
    multisource = True
    general_dir = 'doa_data'
    if multisource:
        general_dir += '_multisource'
    data_dir = 'pseudospectra/'
    normalize = False
    # Load the data from yaml file
    file_no = int(input('File number: ')) # 9 - 28 - 47
    files = os.listdir(os.path.join(general_dir, data_dir))
    if multisource:
        fig = plt.figure(figsize=(12, 5))
        for i in range(3):
            file_name = os.path.join(general_dir, data_dir  + files[file_no + i*15])
            with open(file_name, 'r') as f:
                try:
                    data = yaml.safe_load(f)  # Use safe_load to avoid potential security issues
                except yaml.YAMLError as error:
                    print(error)
            p_dB = np.array(data['p_dB'])
            theta = np.array(data['theta'])
            if multisource:
                gt_angles = np.array(data['obst_position'])
            else:
                gt_angles = data['obst_position']  
            theta_bar = theta[np.argmax(p_dB)]
            print(f"\nDOA: {theta_bar:.1f} [deg]\nGT: {gt_angles} [deg]")
            algo = files[file_no+ i*15][:files[file_no+ i*15].find('_')]
            if algo == 'das':
                title_algo = 'DAS'
            elif algo == 'capon':
                title_algo = 'Capon'
            else:
                title_algo = 'MUSIC'
                p_dB /= 2
            
            if (file_no in [9, 28, 47] and not multisource):
                cs = CubicSpline(theta, p_dB)
                theta_interp = np.arange(-90, 90, 0.1)
                p_dB_interp = cs(theta_interp)
                fig, ax2 = plt.subplots(figsize=(10, 4))
                correction = theta_interp[np.argmax(p_dB_interp)]
                p_dB_interp = cs(theta_interp + correction) - max(cs(theta_interp + correction))
                if normalize:
                    p_dB_interp -= max(p_dB_interp)
                m3dB = theta_interp[(p_dB_interp > -3.2) & (p_dB_interp < -2.8)]
                ax2.plot(theta_interp, p_dB_interp, color='black', linewidth=1)
                print(m3dB)
                ax2.vlines([m3dB[0], m3dB[-1]], -30, -3.01, linestyles='--', colors='b')
                ax2.axhline(-3, linewidth=0.8, color='r')
                ax2.text(-70, -2, '-3 [dB]', color='r')
                ax2.set_xlim(-90, 90)
                ax2.set_ylim(-24, 2)
                ax2.set_xticks(np.linspace(-90, 90, 7))
                ax2.set_xlabel('Angle [deg]')
                ax2.set_ylabel('Magnitude [dB]')
            else:
                ax2 = fig.add_subplot(1, 3, 2 if i==0 else 1 if i==1 else i+1, projection='polar')
                ax2.set_xlim(-np.pi/2*1.1, 1.1*np.pi/2)
                ax2.set_theta_offset(np.pi/2)
                ax2.plot(np.deg2rad(theta), p_dB, color='black', linewidth=1.2)
                ax2.set_xticks(np.linspace(-np.pi/2, np.pi/2, 7))
                ax2.set_ylabel('dB')
                ax2.yaxis.label.set_rotation(0)
                ax2.yaxis.set_label_coords(1.1, 0.13)
                ax2.set_yticks(np.arange(np.floor(np.min(p_dB)), np.ceil(np.max(p_dB)), (np.ceil(np.max(p_dB)) - np.floor(np.min(p_dB)))//3))
                if multisource:
                    ax2.vlines(np.deg2rad(gt_angles), np.floor(np.min(p_dB)), np.ceil(np.max(p_dB)), colors='r', linestyles='-', linewidth=1, label='Ground truth')
                    peaks_pos = signal.argrelmax(p_dB, mode='wrap')
                    peaks_values = p_dB[peaks_pos]
                    peaks_angles = theta[peaks_pos]
                    sorted_peaks_pos = np.argsort(peaks_values)
                    print(peaks_angles[sorted_peaks_pos])
                    ax2.vlines(np.deg2rad(peaks_angles[sorted_peaks_pos[-2:]]), np.floor(np.min(p_dB)), np.ceil(np.max(p_dB)), colors='b', linestyles='--', linewidth=1, label='Highest peaks found')

            ax2.set_title('p($\\theta$) - ' + title_algo)
            if normalize:
                ax2.set_yticks(np.arange(-20, 5, 5))
            ax2.grid(True)
        plt.tight_layout()
        ax2.legend(loc='lower left', bbox_to_anchor=(-1.34, -0.2))
        plt.savefig(str(file_no) + '_multi', dpi=600, transparent=True)
        # plt.show()
