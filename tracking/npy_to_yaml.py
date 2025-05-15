import os
import numpy as np
import yaml

os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())
npy_files = os.listdir('./offsets/')
npy_files = [f for f in npy_files if f.endswith('.npy')]

for npy in npy_files:
    # Load numpy array from file
    array = np.load('./offsets/' + npy)
    print(array.shape)

    data = {
        'offsets': array[:, 1].tolist(),
        'reading_points': array[:, 0].tolist()
    }

    # Save to YAML file
    with open('./offsets/' + npy[:-12] + '.yaml', 'w') as f:
        yaml.dump(data, f)
