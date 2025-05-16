import cv2
from cv2 import aruco
import os
import traceback
import numpy as np
import pyautogui as pag
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

file_name = 'GX010526'

camera_path = './videos/' + file_name + '.mp4'

screen_width, screen_height = pag.size()
robot_id = 0

video_path = camera_path
cap = cv2.VideoCapture(video_path)

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()

frame_count = 0
trajectory = np.zeros((0, 2), dtype=np.float32)
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cv2.imwrite('output.jpg', resized_frame)
            break
        if frame_count % 20 == 0:
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
                        if len(trajectory) > 2:
                            # Draw trajectory
                            for i in range(len(trajectory) - 1):
                                cv2.line(frame, tuple(trajectory[i].astype(int)), tuple(trajectory[i + 1].astype(int)), (0, 255, 0), 2)
                        # Display result
                        resized_frame = cv2.resize(frame, (screen_width, screen_height))
                        cv2.imshow('ArUco Tracker', resized_frame)
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
cap.release()
cv2.destroyAllWindows()