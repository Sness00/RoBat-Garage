import cv2
from cv2 import aruco
import os
import traceback
import yaml
import numpy as np
import pyautogui as pag

# Cross product in 2D
def cross2d(x, y):
    return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]

os.chdir(os.path.dirname(os.path.abspath(__file__)))
screen_width, screen_height = pag.size()
robot_id = 0
arena_w = 100
arena_l = 170
# Load video file
video_path = './videos/cropped_video3.mp4'
cap = cv2.VideoCapture(video_path)

# Camera calibration parameters
yaml_file = "calibration_matrix.yaml"
with open(yaml_file, 'r') as file:
    data = yaml.safe_load(file)

# Load predefined dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()
marker_length = 0.049  # Length of the marker's side in meters

object_points = []  # 3D points in real world space
object_points.append(np.array([[-marker_length/2, marker_length/2, 0], 
                               [marker_length/2, marker_length/2, 0], 
                               [marker_length/2, -marker_length/2, 0], 
                               [-marker_length/2, -marker_length/2, 0]], dtype=np.float32))
object_points = np.array(object_points, dtype='float32')

# Loop through the video
frame_count = 0
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # print(frame_count)
        if frame_count % 20 == 0:
            
            # Convert to grayscale (optional, but recommended)
            # undistort the image
            # h, w = frame.shape[:2]
            # new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(np.array(data['camera_matrix']), np.array(data['dist_coeff']), (w, h), 0.5, (w, h))
            # x, y, w, h = roi
            # # Crop the image to the ROI if needed
            # undistorted_frame = cv2.undistort(frame, np.array(data['camera_matrix']), np.array(data['dist_coeff']), None, new_camera_mtx)
            # undistorted_frame = undistorted_frame[y:y+h, x:x+w]

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect markers
            detector = aruco.ArucoDetector(aruco_dict, parameters)
            corners, ids, _ = detector.detectMarkers(gray)

            # print(f"Frame {frame_count}: Detected IDs: {ids}")
            
            # Draw detected markers
            if ids is not None:
                aruco.drawDetectedMarkers(frame, corners, ids)
                index = np.where(ids == robot_id)[0] # Find the index of the robot marker
                
                arena_mrkr_indx = {
                    '12': np.where(ids == 12)[0], # Find the index of the arena markers
                    '13': np.where(ids == 13)[0], # Find the index of the arena markers
                    '14': np.where(ids == 14)[0], # Find the index of the arena markers
                }
                corners_array = np.squeeze(np.array(corners))
                
                # ret, rvec, tvec = cv2.solvePnP(object_points, corners_array[index], np.array(data['camera_matrix']), np.array(data['dist_coeff']))  
                # if ret:
                #     cv2.drawFrameAxes(resized_frame, np.array(data['camera_matrix']), np.array(data['dist_coeff']), rvec, tvec, marker_length, 1)
                
                # find center of the marker
                center = np.mean(corners_array[index], axis=1)[0]
                # print(center)
                # cv2.circle(frame, center.astype(int), 50, (255, 255, 0), -1)

                mask = np.ones(len(ids), dtype=bool)
                mask[index] = False
                mask[list(arena_mrkr_indx.values())] = False

                obst_ids = ids[mask]
                obst_corners = corners_array[mask]

                obst_centers = np.mean(obst_corners, axis=1)

                distances = np.linalg.norm(obst_centers - center, axis=1)
                if len(distances) > 0:
                    sorted_distances = np.sort(distances)
                    # print(distances)
                    # closest_obstacle_index = np.argmin(distances)
                    # compute angle between the robot and the closest obstacle
                    tl, tr, br, bl = np.squeeze(corners_array[index])
                    D41 = tl - bl
                    D14 = bl - tl
                    D41_normalized = D41 / np.linalg.norm(D41)
                    for sd in sorted_distances:
                        V_marker_space = np.squeeze(obst_centers[np.where(distances == sd)]) - center
                        dot_product = np.dot(D41_normalized, V_marker_space)
                        cross_product = cross2d(V_marker_space, D14)
                        verse = -1 if cross_product > 0 else 1
                        angle = verse*np.arccos(dot_product / (np.linalg.norm(V_marker_space)))
                        # print(f"Angle: {np.rad2deg(angle)}")
                        # print(np.rad2deg(angle))
                        # draw a line from the robot to the closest obstacle
                        if np.abs(angle) <= np.pi/2:
                            # print(f"Angle: {np.rad2deg(angle)}")                
                            closest_obstacle_center = np.squeeze(obst_centers[np.where(distances == sd)])
                            break
                    # print(np.rad2deg(angle))
                    if angle <= np.pi/2:
                        cv2.line(frame, center.astype(int), closest_obstacle_center.astype(int), (0, 51, 20), 2)
                        # print angle and id of nearest obstacle
                        cv2.putText(frame, str(obst_ids[np.where(distances == sd)][0]), closest_obstacle_center.astype(int), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        # print(angle, frame_count)
                        

                        

            #     # You can access marker positions here
            #     for i, corner in enumerate(corners):
            #         print(f"Marker ID: {ids[i][0]}, Corners: {corner}")

            # Display result
            resized_frame = cv2.resize(frame, (screen_width, screen_height))
            cv2.imshow('ArUco Tracker', resized_frame)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print(frame_count, e)
    traceback.print_exc()
cap.release()
cv2.destroyAllWindows()
