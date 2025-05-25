import os
import yaml
import numpy as np
from matplotlib import pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))

data_dir = './trajectories/'

file_names = os.listdir(data_dir)

with open(data_dir + file_names[5], 'r') as file:
    try:
        data = yaml.safe_load(file)  # Use safe_load to avoid potential security issues
    except yaml.YAMLError as error:
        print(f"Error loading YAML file: {error}")

trajectory = data['trajectory']

trajectory = np.array(trajectory)

bottle_radius = 3.2e-2 # [m]
pixel_per_meter = 663.5

trajectory /= pixel_per_meter

bottle_radius_pixels = bottle_radius * pixel_per_meter
print(bottle_radius_pixels)

xedges = int((max(trajectory[:, 0]) - min(trajectory[:, 0]))/bottle_radius)
yedges = int((max(trajectory[:, 1]) - min(trajectory[:, 1]))/bottle_radius)

H, xedges, yedges = np.histogram2d(trajectory[:, 0], trajectory[:, 1], bins=[xedges, yedges], density=True)
H = H.T
X, Y = np.meshgrid(xedges, yedges)

fig = plt.figure()
ax = fig.add_subplot()
ax.pcolormesh(X, Y, H, )
plt.show()