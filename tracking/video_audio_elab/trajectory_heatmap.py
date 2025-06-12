import os
import yaml
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.image import NonUniformImage

os.chdir(os.path.dirname(os.path.abspath(__file__)))

data_dir = './trajectories/'
obst_pos_dir = './analysis/'
file_names = os.listdir(data_dir)
obst_pos_files = os.listdir(obst_pos_dir)

group_indexes = [0, 3, 5, 7, -1]
with open(data_dir + 'conversion_factors.yaml', 'r') as file:
    try:
        data = yaml.safe_load(file)  # Use safe_load to avoid potential security issues
    except yaml.YAMLError as error:
        print(f"Error loading YAML file: {error}")
pixel_per_meter = np.array(data['pixel_to_meters'])
bottle_radius = 3.2e-2 # [m]
fig = plt.figure()
for i in range(len(group_indexes) - 1):
    with open(obst_pos_dir + obst_pos_files[group_indexes[i]], 'r') as file:
            try:
                pos_data = yaml.safe_load(file)  # Use safe_load to avoid potential security issues
            except yaml.YAMLError as error:
                print(f"Error loading YAML file: {error}")
    trajectory = np.empty((0, 2))
    for j, f in enumerate(file_names[group_indexes[i]:group_indexes[i+1]]):
        with open(data_dir + f, 'r') as file:
            try:
                data = yaml.safe_load(file)  # Use safe_load to avoid potential security issues
            except yaml.YAMLError as error:
                print(f"Error loading YAML file: {error}")
        traj = np.array(data['trajectory'])
        norm_x = min(traj[:, 0])
        norm_y = min(traj[:, 1])
        traj[:, 0] -= norm_x  # Normalize x-coordinates
        traj[:, 1] -= norm_y  # Normalize y-coordinates
        traj /= pixel_per_meter[j + group_indexes[i]]  # Convert to meters
        trajectory = np.vstack((trajectory, traj))   
    obst_positions = np.array(pos_data['obstacles_positions'])
    obst_positions[:, 0] -= norm_x
    obst_positions[:, 1] -= norm_y
    obst_positions /= pixel_per_meter[group_indexes[i]]
    # obst_positions = np.flipud(obst_positions)
    # bottle_radius_pixels = bottle_radius * pixel_per_meter
    xedges = int((max(trajectory[:, 0]) - min(trajectory[:, 0]))/bottle_radius)

    yedges = int((max(trajectory[:, 1]) - min(trajectory[:, 1]))/bottle_radius)

    H, xedges, yedges = np.histogram2d(trajectory[:, 0], trajectory[:, 1], bins=[xedges, yedges])
    H[H > 20] = 20
    # Normalize the histogram
    # H = H / np.sum(H) if np.sum(H) > 0 else H  # Avoid division by zero
    # flip the histogram to match the orientation of the trajectory
    H = H.T
    H = np.flipud(H)
    ax = fig.add_subplot(2, 3, i+1, aspect='equal')
    X, Y = np.meshgrid(xedges, yedges)
    mesh = ax.pcolormesh(X, Y, H)
    fig.colorbar(mesh, ax=ax, orientation='horizontal')    
    ax.scatter(obst_positions[:, 0], max(trajectory[:, 1]) - obst_positions[:, 1], 100, 'r', '+')
    ax.set_title(f'Configuration {i+1}')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1.55)
    # ax.set_aspect('equal', adjustable='box')
    

data_dir = './trajectories_control/'
file_names = os.listdir(data_dir)
with open(data_dir + 'conversion_factors.yaml', 'r') as file:
    try:
        data = yaml.safe_load(file)  # Use safe_load to avoid potential security issues
    except yaml.YAMLError as error:
        print(f"Error loading YAML file: {error}")
pixel_per_meter = np.array(data['pixel_to_meters'])
for k, f in enumerate(file_names[:-1]):
    with open(data_dir + f, 'r') as file:
        try:
            data = yaml.safe_load(file)  # Use safe_load to avoid potential security issues
        except yaml.YAMLError as error:
            print(f"Error loading YAML file: {error}")
    traj = np.array(data['trajectory'])  # Assuming the trajectory is the first key in the dictionary
    traj[:, 0] -= min(traj[:, 0])  # Normalize x-coordinates
    traj[:, 1] -= min(traj[:, 1])  # Normalize y-coordinates
    traj /= pixel_per_meter[k]  # Convert to meters
    trajectory = traj

    xedges = int((max(trajectory[:, 0]) - min(trajectory[:, 0]))/bottle_radius)

    yedges = int((max(trajectory[:, 1]) - min(trajectory[:, 1]))/bottle_radius)

    H, xedges, yedges = np.histogram2d(trajectory[:, 0], trajectory[:, 1], bins=[xedges, yedges])
    H[H > 20] = 20
    # Normalize the histogram
    # H = H / np.sum(H) if np.sum(H) > 0 else H  # Avoid division by zero
    # flip the histogram to match the orientation of the trajectory
    H = H.T
    H = np.flipud(H)
    ax = fig.add_subplot(2, 3, k + i + 2, aspect='equal')
    X, Y = np.meshgrid(xedges, yedges)
    mesh = ax.pcolormesh(X, Y, H)
    fig.colorbar(mesh, ax=ax, orientation='horizontal') 
    ax.set_title(f'No obstacles {k+1}')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1.55)
    # ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.suptitle('Trajectory Heatmaps', fontsize=16)
plt.show()
