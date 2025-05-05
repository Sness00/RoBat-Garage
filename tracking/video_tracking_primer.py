import cv2
import cv2.aruco as aruco
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load video file
video_path = './2_7K_1.mp4'
cap = cv2.VideoCapture(video_path)

# Load predefined dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()

# Loop through the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale (optional, but recommended)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(gray)

    print(f"Detected IDs: {ids}")

    # # Draw detected markers
    # if ids is not None:
    #     aruco.drawDetectedMarkers(frame, corners, ids)

    #     # You can access marker positions here
    #     for i, corner in enumerate(corners):
    #         print(f"Marker ID: {ids[i][0]}, Corners: {corner}")

    # Display result
    # cv2.imshow('ArUco Tracker', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
