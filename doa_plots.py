import os
import yaml
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    plt.rcParams['text.usetex'] = True
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    # Set the path to the directory containing the .npy files
    multisource = True
    general_dir = 'doa_data'
    if multisource:
        general_dir += '_multisource'
    data_dir = 'pseudospectra/'
    normalize = True
    # Load the data from yaml file
    files = os.listdir(os.path.join(general_dir, data_dir))
    file_name = os.path.join(general_dir, data_dir  + files[40])
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
    print(f"Ground Truth Angles: {gt_angles}")
    # p_dB = np.load(os.path.join(general_dir, data_dir  + 'music_20250429_18-06-03_0.npy'))
    if normalize:
        p_dB -= max(p_dB)
    # theta = np.linspace(-90, 90, p_dB.shape[0])    
    theta_bar = theta[np.argmax(p_dB)]
    print(f"\nDOA: {theta_bar:.1f} degrees")
    
    # Create a scatter plot of the DOA data
    fig, ax2 = plt.subplots(subplot_kw={'projection': 'polar'})
    ax2.set_title('p($\\theta$)', fontsize=20)
    ax2.plot(np.deg2rad(theta), p_dB, color='black', linewidth=1.5)
    if multisource:
        for angle in gt_angles:
            ax2.axvline(np.deg2rad(angle), color='red', linestyle='dashed', label='Ground Truth')
    else:
        ax2.axvline(obst_position := np.deg2rad(gt_angles), color='red', linestyle='dashed', label='Ground Truth')
    for label in ax2.get_yticklabels():
        label.set_fontsize(16)
    # ax2.axvline(np.deg2rad(theta_bar), color='g', linestyle='dashed')
    # ax2.set_title('Pseudospectrum')
    ax2.set_xlim(-np.pi/2, np.pi/2)
    ax2.set_theta_offset(np.pi/2)
    ax2.set_rlabel_position(0)
    ax2.text(np.deg2rad(-90), 5, 'dB', 
        ha='center', va='center', rotation=0, fontdict={'fontsize': 16})
    if normalize:
        ax2.set_yticks([-20, -10, 0])
    ax2.set_xticks(np.deg2rad([-80, -60, -40, -20, 0, 20, 40, 60, 80]))
    for label in ax2.get_xticklabels():
        label.set_fontsize(16) 
    ax2.legend()
    ax2.grid(True)
   
    plt.show()
