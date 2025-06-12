import os
import yaml
import numpy as np
from matplotlib import pyplot as plt


def compute_stats(x, y, bins):
    means = []
    stds = []
    for i in range(len(bins)-1):
        mask = (x >= bins[i]) & (x < bins[i+1])
        if any(mask):
            means.append(np.mean(y[mask]))
            stds.append(np.std(y[mask]))
        else:
            means.append(np.nan)
            stds.append(np.nan)
    return np.array(means), np.array(stds)

os.chdir(os.path.dirname(os.path.abspath(__file__)))

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

results_dir = './plots/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

data_dir = './non_blind_analysis/'
# data_dir = './analysis/'

data_files = os.listdir(data_dir)
# Filter out non-YAML files
data_files = [f for f in data_files if f.endswith('.yaml')]

obst_distances = []
distance_errors = []
obst_angles = []
angle_errors = []

for file_name in data_files:
    with open(data_dir + file_name, "r") as file:
        try:
            data = yaml.safe_load(file)  # Use safe_load to avoid potential security issues
        except yaml.YAMLError as error:
            print(f"Error loading YAML file: {error}")

        # Extracting the data
        obst_distances.append(data['obstacle_distances'])
        distance_errors.append(data['distance_errors'])
        obst_angles.append(data['obstacle_angles'])
        angle_errors.append(data['angle_errors'])

# Flatten the lists of lists into single lists
obst_distances = [item for sublist in obst_distances for item in sublist]
distance_errors = [item for sublist in distance_errors for item in sublist]
obst_angles = [item for sublist in obst_angles for item in sublist]
angle_errors = [item for sublist in angle_errors for item in sublist]

# Print the number of data points
print(f"Number of data points: {len(obst_distances)}")

# Convert lists to numpy arrays for easier manipulation
obst_distances = np.array(obst_distances)
distance_errors = np.array(distance_errors)
obst_angles = np.array(obst_angles)
angle_errors = np.array(angle_errors)

distance_errors[(obst_distances > 100) | (obst_distances < 12.5)] = np.nan
obst_distances[(obst_distances > 100) | (obst_distances < 12.5)] = np.nan
obst_distances[(distance_errors > 10) | (distance_errors < -10)] = np.nan
distance_errors[(distance_errors > 10) | (distance_errors < -10)] = np.nan
obst_angles[distance_errors == np.nan] = np.nan
angle_errors[distance_errors == np.nan] = np.nan
distance_errors = distance_errors[~ np.isnan(obst_distances)]
obst_angles = obst_angles[~ np.isnan(obst_distances)]
angle_errors = angle_errors[~ np.isnan(obst_distances)]
obst_distances = obst_distances[~ np.isnan(obst_distances)]
print(f"Number of data points after outliers elimination: {len(obst_distances)}")


dst_err_mean = np.mean(distance_errors)
dst_err_median = np.median(distance_errors)
dst_err_std = np.std(distance_errors)

ang_err_mean = np.mean(angle_errors)
ang_err_median = np.median(angle_errors)
ang_err_std = np.std(angle_errors)

angle_bins = np.arange(-90, 91, 5)  # Create bins every 5 deg
ang_bin_centers = (angle_bins[:-1] + angle_bins[1:]) / 2
ang_mean, ang_std = compute_stats(obst_angles, angle_errors, angle_bins)


distance_bins = np.arange(10, 101, 5)  # Create bins every 5 deg
dist_bin_centers = (distance_bins[:-1] + distance_bins[1:]) / 2
dist_mean, dist_std = compute_stats(obst_distances, distance_errors, distance_bins)

ang_mean, ang_std = compute_stats(obst_angles, angle_errors, angle_bins)
dist_mean, dist_std = compute_stats(obst_distances, distance_errors, distance_bins)

dist_vs_ang_mean, dist_vs_ang_std = compute_stats(obst_angles, distance_errors, angle_bins)
ang_vs_dist_mean, ang_vs_dist_std = compute_stats(obst_distances, angle_errors, distance_bins)

# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 2)
# plt.plot(ang_bin_centers, ang_mean, linewidth=2, color='red')
# plt.fill_between(ang_bin_centers, ang_mean-ang_std, ang_mean+ang_std, color='red', alpha=0.2,linewidth=2)
# plt.title('Angle Errors vs Ostacle Angles', fontsize=20)
# plt.xlabel('Obstacle Angles [deg]', fontsize=16)
# plt.ylabel('Angle Errors [deg]', fontsize=16)
# plt.xlim(-100, 100)
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# plt.grid()

# plt.subplot(1, 2, 1)
# plt.plot(dist_bin_centers, dist_mean, linewidth=2, color='blue')
# plt.fill_between(dist_bin_centers, dist_mean-dist_std, dist_mean+dist_std, color='blue', alpha=0.2,linewidth=2)
# plt.title('Distance Errors vs Obstacle Distances', fontsize=20)
# plt.xlabel('Obstacle Distances [cm]', fontsize=16)
# plt.ylabel('Distance Errors [cm]', fontsize=16)
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# plt.grid()
# plt.tight_layout()
# # plt.savefig(data_dir + 'line', dpi=600, transparent=True)

# # Create a scatter plot for distance errors
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# plt.scatter(obst_distances, distance_errors, color='blue', alpha=0.5)
# plt.title('Distance Errors vs Obstacle Distances', fontsize=20)
# plt.xlabel('Obstacle Distances [cm]', fontsize=16)
# plt.ylabel('Distance Errors [cm]', fontsize=16)
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# plt.grid()

# plt.subplot(1, 2, 2)
# # Create a scatter plot for angle errors
# plt.scatter(obst_angles, angle_errors, color='red', alpha=0.5)
# plt.title('Angle Errors vs Obstacle Angles', fontsize=20)
# plt.xlabel('Obstacle Angles [deg]', fontsize=16)
# plt.ylabel('Angle Errors [deg]', fontsize=16)
# plt.xlim(-100, 100)
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# plt.grid()
# plt.tight_layout()
# # plt.savefig(data_dir + 'scatter', dpi=600, transparent=True)

# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 2)
# plt.plot(ang_bin_centers, dist_vs_ang_mean, linewidth=2, color='red')
# plt.fill_between(ang_bin_centers, dist_vs_ang_mean-dist_vs_ang_std, dist_vs_ang_mean+dist_vs_ang_std, color='red', alpha=0.2,linewidth=2)
# plt.title('Distance Error vs Ostacle Angles', fontsize=20)
# plt.xlabel('Obstacle Angles [deg]', fontsize=16)
# plt.ylabel('Distance Errors [cm]', fontsize=16)
# plt.grid()

# plt.subplot(1, 2, 1)
# plt.plot(dist_bin_centers, ang_vs_dist_mean, linewidth=2, color='blue')
# plt.fill_between(dist_bin_centers, ang_vs_dist_mean-ang_vs_dist_std, dist_mean+ang_vs_dist_std, color='blue', alpha=0.2,linewidth=2)
# plt.title('Angle Errors vs Obstacle Distances', fontsize=20)
# plt.xlabel('Obstacle Distances [cm]', fontsize=16)
# plt.ylabel('Angle Errors [deg]', fontsize=16)
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# plt.grid()
# plt.tight_layout()

# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# plt.scatter(obst_distances, angle_errors, color='blue', alpha=0.5)
# plt.title('Angle Error vs Obstacle Distances', fontsize=20)
# plt.xlabel('Obstacle Distances [cm]', fontsize=16)
# plt.ylabel('Angle Error [deg]', fontsize=16)
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# plt.grid()

# plt.subplot(1, 2, 2)
# # Create a scatter plot for angle errors
# plt.scatter(obst_angles, distance_errors, color='red', alpha=0.5)
# plt.title('Distance Errors vs Obstacle Angles', fontsize=20)
# plt.xlabel('Obstacle Angles [deg]', fontsize=16)
# plt.ylabel('Distance Errors [cm]', fontsize=16)
# # plt.xlim(-100, 100)
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# plt.grid()
# plt.tight_layout()

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
# Violin plot for distance errors, blue
vp1 = plt.violinplot(distance_errors, np.ones((1)), showmedians=True, showextrema=False)
for body in vp1['bodies']:
    body.set_facecolor('blue')
    body.set_alpha(0.5)
# Set means line color to black
if 'cmedians' in vp1:
    vp1['cmedians'].set_color('black')
plt.text(0.2, 0.8, f'Mean: {dst_err_mean:.2f} [cm]\nMedian: {dst_err_median:.2f} [cm]\nStd: {dst_err_std:.2f} [cm]',
         horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=14)
# Hide the x-axis label
plt.xticks([])
plt.title('Distance error distribution', fontsize=20)
plt.ylabel('Distance Errors [cm]', fontsize=16)
# plt.legend()
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)

plt.subplot(1, 2, 2)
# Violin plot for angle errors, red
vp2 = plt.violinplot(angle_errors, np.ones((1)), showmedians=True, showextrema=False)
for body in vp2['bodies']:
    body.set_facecolor('red')
    body.set_alpha(0.5)
# Set means line color to black
if 'cmedians' in vp2:
    vp2['cmedians'].set_color('black')
plt.text(0.2, 0.8, f'Mean: {ang_err_mean:.2f} [deg]\nMedian: {ang_err_median:.2f} [deg]\nStd: {ang_err_std:.2f} [deg]',
         horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=14)
plt.title('Angle error distribution', fontsize=20)
plt.ylabel('Angle Errors [deg]', fontsize=16)
plt.xticks([])
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.tight_layout()
plt.savefig(data_dir + 'violin', dpi=600, transparent=True)

# plt.savefig(results_dir + 'all_runs.png', dpi=300, bbox_inches='tight')


# colors = ['#0000FF', '#00FF00', '#FFFF00', '#FF8000', '#FF0000', '#800080']

# bins = np.array([-90, -60, -30, 0, 30, 60, 90])
# indices = np.digitize(obst_angles, bins, right=False) - 1
# indices = np.clip(indices, 0, len(colors) - 1)
# associated_colors = np.array(colors)[indices]

# # Create a scatter plot for angle errors vs distance errors
# plt.figure(figsize=(16, 9))
# plt.scatter(angle_errors, distance_errors, color=associated_colors, alpha=0.5)
# plt.title('Distance Errors vs Angle Errors', fontsize=20)
# plt.xlabel('Angle Errors [deg]', fontsize=16)
# plt.ylabel('Distance Errors [cm]', fontsize=16)
# plt.grid()
# # Legend for the colors
# for i, color in enumerate(colors):
#     plt.scatter([], [], color=color, label=fr'{bins[i]} [deg] $\leq \theta_G$$_T <$  {bins[i+1]} [deg]')
# plt.xlim(-180, 180)
# plt.yticks(fontsize=16)
# plt.xticks(fontsize=16)
# # plt.ylim(-90, 30)
# plt.legend(fontsize=16)
# plt.tight_layout()
# # plt.savefig(data_dir + 'err_vs_err', dpi=600, transparent=True)
plt.show()
