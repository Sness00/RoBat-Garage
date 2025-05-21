import os
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    plt.rcParams['text.usetex'] = True
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    # Set the path to the directory containing the .npy files
    data_dir = './pseudospectra/'
    
    # Load the data from the .npy files
    p_dB = np.load(data_dir  + 'music_20250429_18-00-41_-60.npy')
    theta = np.linspace(-90, 90, p_dB.shape[0])    
    theta_bar = theta[np.argmax(p_dB)]
    print(f"\nDOA: {theta_bar:.1f} degrees")
    
    # Create a scatter plot of the DOA data
    fig, ax2 = plt.subplots(subplot_kw={'projection': 'polar'})
    ax2.plot(np.deg2rad(theta), p_dB)
    ax2.axvline(np.deg2rad(theta_bar), color='g', linestyle='dashed')
    ax2.set_title('Pseudospectrum')
    ax2.set_theta_offset(np.pi/2)
    ax2.set_xlim(-np.pi/2, np.pi/2)
    ax2.set_xticks(np.deg2rad([-80, -60, -40, -20, 0, 20, 40, 60, 80]))    
    ax2.grid(True)
    plt.show()
