import os
import yaml
import numpy as np
from matplotlib import pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))

results_dir = './plots/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

data_dir = './analysis/'

data_files = os.listdir(data_dir)
# Filter out non-YAML files
data_files = [f for f in data_files if f.endswith('.yaml')]

obst_distances = []
distance_errors = []
obst_angles = []
angle_errors = []

for file_name in data_files:
    with open('./analysis/' + file_name, "r") as file:
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
dst_err_mean = np.mean(distance_errors)
dst_err_std = np.std(distance_errors)
obst_angles = np.array(obst_angles)
angle_errors = np.array(angle_errors)
ang_err_mean = np.mean(angle_errors)
ang_err_std = np.std(angle_errors)

# Create a scatter plot for distance errors
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.scatter(obst_distances, distance_errors, color='blue', alpha=0.5)
plt.title('Distance Errors vs Obstacle Distances')
plt.xlabel('Obstacle Distances (cm)')
plt.ylabel('Distance Errors (cm)')
plt.grid()

plt.subplot(2, 2, 2)
# Create a scatter plot for angle errors
plt.scatter(obst_angles, angle_errors, color='red', alpha=0.5)
plt.title('Angle Errors vs Obstacle Angles')
plt.xlabel('Obstacle Angles (degrees)')
plt.ylabel('Angle Errors (degrees)')
plt.grid()

plt.subplot(2, 2, 3)
# Violin plot for distance errors, blue
vp1 = plt.violinplot(distance_errors, np.ones((1)), showmeans=True, showextrema=False)
for body in vp1['bodies']:
    body.set_facecolor('blue')
    body.set_alpha(0.5)
# Set means line color to black
if 'cmeans' in vp1:
    vp1['cmeans'].set_color('black')
plt.text(0.2, 0.2, f'Mean: {dst_err_mean:.2f}cm\nStd: {dst_err_std:.2f}cm', 
         horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
# Hide the x-axis label
plt.xticks([])
plt.title('Distance Error')
plt.ylabel('Distance Errors (cm)')
plt.grid()

plt.subplot(2, 2, 4)
# Violin plot for angle errors, red
vp2 = plt.violinplot(angle_errors, np.ones((1)), showmeans=True, showextrema=False)
for body in vp2['bodies']:
    body.set_facecolor('red')
    body.set_alpha(0.5)
# Set means line color to black
if 'cmeans' in vp2:
    vp2['cmeans'].set_color('black')
plt.text(0.2, 0.2, f'Mean: {ang_err_mean:.2f}°\nStd: {ang_err_std:.2f}°', 
         horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
plt.title('Angle Error')
plt.ylabel('Angle Errors (degrees)')
plt.xticks([])
plt.grid()
plt.tight_layout()
# Save the figure
plt.savefig(results_dir + 'all_runs.png', dpi=300, bbox_inches='tight')
plt.show()
