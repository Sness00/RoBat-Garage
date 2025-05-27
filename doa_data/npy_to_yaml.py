import os
import numpy as np
import yaml

os.chdir(os.path.dirname(os.path.abspath(__file__)))

npy_files = os.listdir('./pseudospectra/')
npy_files = [f for f in npy_files if f.endswith('.npy')]

for npy in npy_files:
    # Load numpy array from file
    array = np.load('./pseudospectra/' + npy)
    print(array.shape)
    data = {
        'bw': (20e3, 60e3),
        'd': 2.7e-3,
        'fs': 176400,
        'method': npy[:npy.find('_')],
        'nch': 8,
        'obst_position': np.array(int(npy[-7 + npy[-8:].find('_'):npy.find('.')])).tolist(),
        'p_dB': array.tolist(),
        'theta': np.linspace(-90, 90, len(array)).tolist()
    }
    print(str(data['method']))
    # Save to YAML file
    with open('./pseudospectra/' + npy[:21 if str(data['method']) == 'das' else 23] + '.yaml', 'w') as f:
        yaml.dump(data, f)
