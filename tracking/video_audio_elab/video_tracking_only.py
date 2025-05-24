import cv2
from cv2 import aruco
import os
import traceback
import numpy as np
import pyautogui as pag
import sys
import yaml

os.chdir(os.path.dirname(os.path.abspath(__file__)))
save_dir = './trajectories/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()



screen_width, screen_height = pag.size()
robot_id = 0

camera_path = './videos/'
file_names = os.listdir(camera_path)
file_names = [f for f in file_names if f.endswith('.MP4')]
for file_name in file_names:
    print(file_name[:-4])
    video_path = camera_path + file_name
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    trajectory = np.zeros((0, 2), dtype=np.float32)
    try:
        while cap.isOpened():
            print(frame_count)
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % 10 == 0:
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
                    if ids is not None:
                        corners_array = np.squeeze(np.array(corners))
                        try:
                            index = np.where(ids == robot_id)[0] # Find the index of the robot marker
                            if len(index) == 0:
                                raise ValueError('Robot marker not found')
                            center = np.mean(corners_array[index], axis=1)[0]
                            trajectory = np.append(trajectory, np.array([[center[0], center[1]]]), axis=0)
                            # if len(trajectory) > 2:
                            #     # Draw trajectory
                            #     for i in range(len(trajectory) - 1):
                            #         cv2.line(frame, tuple(trajectory[i].astype(int)), tuple(trajectory[i + 1].astype(int)), (0, 255, 0), 2)
                            # Display result
                            # resized_frame = cv2.resize(frame, (screen_width, screen_height))
                            # cv2.imshow('ArUco Tracker', resized_frame)
                        except ValueError as e:
                            print(f"Error: {e}")
                            pass
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    traceback.print_exc()
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(frame_count, e)
        traceback.print_exc()
    # Save trajectory to YAML file
    output_yaml = save_dir + file_name[:-4] + '_trajectory.yaml'
    with open(output_yaml, 'w') as f:
        yaml.dump({'trajectory': trajectory.tolist()}, f)
    print(f"Trajectory saved to {output_yaml}")
    cap.release()
cv2.destroyAllWindows()