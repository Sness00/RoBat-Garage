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

def get_offset_point(center, top_left, top_right, offset=100):
    """
    Compute a point 100 pixels in the normal direction from the top edge
    of the ArUco marker.
    
    Parameters:
        center (tuple): (cx, cy), the center of the marker.
        top_left (tuple): (x, y) of the top-left corner.
        top_right (tuple): (x, y) of the top-right corner.
        offset (float): Distance in pixels to move from center (default 100).
    
    Returns:
        (x, y): The new 2D coordinates offset from the center.
    """
    # Convert to numpy arrays
    pt1 = np.array(top_left, dtype=np.float32)
    pt2 = np.array(top_right, dtype=np.float32)
    center = np.array(center, dtype=np.float32)

    # Direction vector of the top edge
    edge_vec = pt2 - pt1
    edge_vec /= np.linalg.norm(edge_vec)  # Normalize

    # Rotate 90 degrees counter-clockwise to get outward normal
    normal_vec = np.array([-edge_vec[1], edge_vec[0]])

    # Compute the new point
    new_point = center + offset * normal_vec

    return new_point

def shift_toward_point(points, p1, shift_cm, px_per_cm):
    """
    Shift p2 toward p1 by shift_cm (in cm), and return new point and distance.
    
    Parameters:
        p1 (tuple): First point (x1, y1)
        p2 (tuple): Second point (x2, y2)
        shift_cm (float): Distance to shift p2 toward p1 (in cm)
        px_per_cm (float): Conversion factor from cm to pixels

    Returns:
        shifted_p2 (tuple): New coordinates of p2 after shifting
        new_distance (float): Distance from p1 to shifted_p2 (in cm)
    """
    coordinates = []
    distances = []
    p1 = np.array(p1, dtype=np.float32)
    for p2 in points:
        # Convert to numpy arrays       
        p2 = np.array(p2, dtype=np.float32)

        # Compute the direction vector from p2 to p1
        direction = p1 - p2
        distance_px = np.linalg.norm(direction)

        if distance_px == 0:
            raise ValueError("Points are identical; cannot compute direction.")

        # Normalize direction and compute shift in pixels
        direction_normalized = direction / distance_px
        shift_px = shift_cm * px_per_cm

        # Shift p2 toward p1
        shifted_p2 = p2 + direction_normalized * shift_px

        # Compute new distance (in pixels), then convert to cm
        new_distance_px = np.linalg.norm(p1 - shifted_p2)
        new_distance_cm = new_distance_px / px_per_cm
        coordinates.append(shifted_p2)
        distances.append(new_distance_cm)

    return np.array(coordinates), np.array(distances)

os.chdir(os.path.dirname(os.path.abspath(__file__)))
screen_width, screen_height = pag.size()
robot_id = 0
arena_w = 1
arena_l = 1.7
# Load video file
video_path = './videos/full_video_obstacles.mp4'
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

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect markers
            detector = aruco.ArucoDetector(aruco_dict, parameters)
            corners, ids, _ = detector.detectMarkers(gray)

            # print(f"Frame {frame_count}: Detected IDs: {ids}")
            
            # Draw detected markers
            if ids is not None:
                corners_array = np.squeeze(np.array(corners))
                aruco.drawDetectedMarkers(frame, corners, ids)
                index = np.where(ids == robot_id)[0] # Find the index of the robot marker
                # print(index)
                arena_mrkr_indx = {
                    '12': np.where(ids == 12)[0], # Find the index of the arena markers
                    '13': np.where(ids == 13)[0], # Find the index of the arena markers
                    '14': np.where(ids == 14)[0], # Find the index of the arena markers
                } # bottom left of 12, top left of 13, top right of 14

                corners_12 = corners_array[arena_mrkr_indx['12']]
                corners_13 = corners_array[arena_mrkr_indx['13']]
                corners_14 = corners_array[arena_mrkr_indx['14']]
                pixel_per_meters = np.mean([np.linalg.norm(corners_12[:, 3] - corners_13[:, 0], axis=1)/arena_w, np.linalg.norm(corners_13[:, 0] - corners_14[:, 1], axis=1)/arena_l])
                # print(pixel_per_meters)
                
                
                # ret, rvec, tvec = cv2.solvePnP(object_points, corners_array[index], np.array(data['camera_matrix']), np.array(data['dist_coeff']))                
                # if ret:
                # R = cv2.Rodrigues(rvec)[0] 
                # cv2.drawFrameAxes(frame, np.array(data['camera_matrix']), np.array(data['dist_coeff']), rvec, tvec, marker_length, 2)
                # T_camera_marker = np.block([
                #                         [R, tvec],
                #                         [np.zeros((1, 3)), 1]
                #                     ])
                # T_translation_marker = np.array([
                #                             [1, 0, 0, 0],
                #                             [0, 1, 0, -0.5],
                #                             [0, 0, 1, 0],
                #                             [0, 0, 0, 1]
                #                         ])
                # T_new = T_camera_marker @ T_translation_marker
                # mic_position = T_new[:3, 3]
                # mic_position_pixels = mic_position[:2] * pixel_per_meters * -0.5
                # print(mic_position_pixels)                   
                # mic_position_pixels = cv2.projectPoints(mic_position, rvec, tvec, np.array(data['camera_matrix']), np.array(data['dist_coeff']))[0]
                # mic_position_pixels = np.squeeze(mic_position_pixels.reshape(-1, 2))
                # print(mic_position_pixels)
                # center = center[:2]
                # find center of the marker
                center = np.mean(corners_array[index], axis=1)[0]
                tl, tr, br, bl = np.squeeze(corners_array[index])
                mic_positions = np.astype(get_offset_point(center, tl, tr, offset=-pixel_per_meters*0.055), np.int32)
                
                # print(center[1]/center[0])
                # center = mic_position_pixels.reshape(-1, 2)[0]
                
                # print(center)

                # rotation_angle = 0  # Replace with actual rotation angle

                # # Convert angle to radians
                # theta_rad = np.radians(rotation_angle)

                # # Compute the translation vector (100 pixels forward along the X-axis)
                # translation = np.array([100, 0])  # Moving along X direction

                # # Compute the rotation matrix for the marker's reference frame
                # rotation_matrix = np.array([
                #     [np.cos(theta_rad), -np.sin(theta_rad)],
                #     [np.sin(theta_rad), np.cos(theta_rad)]
                # ])

                # # Transform the translation vector into global coordinates
                # global_translation = rotation_matrix @ translation

                # # Compute the new center position
                # center = center + global_translation
                # print(center)
                # cv2.circle(frame, center.astype(int), 50, (255, 255, 0), -1)

                mask = np.ones(len(ids), dtype=bool)
                mask[index] = False
                mask[list(arena_mrkr_indx.values())] = False

                obst_ids = ids[mask]
                obst_corners = corners_array[mask]

                obst_centers = np.mean(obst_corners, axis=1)
                for c in obst_centers:
                    cv2.circle(frame, c.astype(int), int(0.032*pixel_per_meters), (0, 0, 255), -1)

                # distances = np.linalg.norm(obst_centers - mic_positions, axis=1) - pixel_per_meters*0.032
                obstacles, distances = shift_toward_point(obst_centers, mic_positions, 3.2, pixel_per_meters/100)
                if len(distances) > 0:
                    # print(distances)
                    # closest_obstacle_index = np.argmin(distances)
                    # compute angle between the robot and the closest obstacle
                    D41 = tl - bl
                    D14 = bl - tl
                    D41_normalized = D41 / np.linalg.norm(D41)
                    sorted_distances = np.sort(distances)
                    found_nearest = False
                    for sd in sorted_distances:
                        V_marker_space = np.squeeze(obstacles[np.where(distances == sd)]) - mic_positions
                        dot_product = np.dot(D41_normalized, V_marker_space)
                        cross_product = cross2d(V_marker_space, D14)
                        verse = -1 if cross_product > 0 else 1
                        angle = verse*np.arccos(dot_product / (np.linalg.norm(V_marker_space)))
                        # print(f"Angle: {np.rad2deg(angle)}")
                        # print(np.rad2deg(angle))
                        # draw a line from the robot to the closest obstacle
                        if np.abs(angle) <= np.pi/2:
                            # print(f"Angle: {np.rad2deg(angle)}")                
                            closest_obstacle = np.squeeze(obstacles[np.where(distances == sd)])
                            found_nearest = True
                            break
                    # print(np.rad2deg(angle))
                    if (found_nearest and angle <= np.pi/2):
                        # print distance
                        print(f"Distance: {sd} cm")
                        cv2.line(frame, mic_positions, closest_obstacle.astype(int), (255, 51, 20), 2)
                        # print angle and id of nearest obstacle
                        # cv2.putText(frame, str(obst_ids[np.where(distances == sd)][0]), closest_obstacle.astype(int), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
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
