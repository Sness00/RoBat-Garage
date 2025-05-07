import os
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

offsets = np.load('20250507_18-18-43_offsets.npy')

print(offsets.shape)
print(max(offsets))