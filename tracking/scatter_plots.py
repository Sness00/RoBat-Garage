import os
import yaml
import numpy as np
from matplotlib import pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))

file_name = ''

with open(os.path.join('./analysis', file_name + '.yml'), "r") as file:
    try:
        data = yaml.safe_load(file)  # Use safe_load to avoid potential security issues
    except yaml.YAMLError as error:
        print(f"Error loading YAML file: {error}")

# Extracting the data
obst_distances = data['obstacle_distances']
distance_errors = data['distance_errors']
obst_angles = data['obstacle_angles']
angle_errors = data['angle_errors']

# Convert lists to numpy arrays for easier manipulation
obst_distances = np.array(obst_distances)
print(len(obst_distances))
distance_errors = np.array(distance_errors)
obst_angles = np.array(obst_angles)
angle_errors = np.array(angle_errors)

# Create a scatter plot for distance errors
plt.figure()
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
plt.title('Angle Error')
plt.ylabel('Angle Errors (degrees)')
plt.grid()
plt.tight_layout()
plt.show()

