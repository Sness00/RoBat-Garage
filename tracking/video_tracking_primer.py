import cv2
from cv2 import aruco
import os
import traceback
import yaml
import numpy as np
import pyautogui as pag
import sys
import io
import ffmpeg
import soundfile as sf
import librosa
from sonar import sonar
from scipy import signal
from das_v2 import das_filter
from capon import capon_method
from matplotlib import pyplot as plt

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

def insert_between_large_diffs(arr):
    result = []

    for i in range(len(arr) - 1):
        result.append(arr[i])
        diff = arr[i+1] - arr[i]

        if abs(diff) > 10:
            # Insert two evenly spaced values between arr[i] and arr[i+1]
            step = diff / 3
            result.append(arr[i] + step)
            result.append(arr[i] + 2 * step)

    result.append(arr[-1])  # Add the last element
    return np.array(result)

def pow_two_pad_and_window(vec, show=False):
    window = signal.windows.tukey(len(vec), alpha=0.3)
    windowed_vec = vec * window
    padded_windowed_vec = np.pad(windowed_vec, (0, 2**int(np.ceil(np.log2(len(windowed_vec)))) - len(windowed_vec)))
    if show:
        dur = len(padded_windowed_vec) / fs
        t = np.linspace(0, dur, len(padded_windowed_vec))
        plt.figure()
        plt.plot(t, padded_windowed_vec)
        plt.show()
    return padded_windowed_vec/max(padded_windowed_vec)*0.8

os.chdir(os.path.dirname(os.path.abspath(__file__)))

camera_path = './videos/GX010518.mp4'
robot_path = './audio/20250514_17-07-06.wav'
offsets = np.load('./offsets/20250514_17-07-06_offsets.npy')
video_fps = 60
screen_width, screen_height = pag.size()
robot_id = 0
arena_w = 1
arena_l = 1.7
# Load video file
video_path = camera_path
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
try:
    robot_audio, fs = sf.read(robot_path)
    robot_audio = robot_audio[:, 0]
    print( 'Robot audio duration: %.1f [s]' % (len(robot_audio)/fs))
    # Run ffmpeg to extract audio and pipe as WAV
    out, _ = (
        ffmpeg
        .input(camera_path)
        .output('pipe:', format='wav', acodec='pcm_s16le')
        .run(capture_stdout=True, capture_stderr=True)
    )

    # Load audio from bytes using soundfile
    camera_audio, sr = librosa.load(io.BytesIO(out), sr=fs, mono=True)
    print( 'Camera audio duration: %.1f [s]' % (len(camera_audio)/fs))

    xcorr = np.roll(signal.correlate(camera_audio, robot_audio, mode='same'), -len(robot_audio) // 2)
    index = np.argmax(np.abs(xcorr))
    start_frame = int(index / sr * video_fps)
    print('Start frame: %d' % start_frame)
    video_frames = np.astype((offsets[:, 0]) / fs * video_fps + start_frame, np.int32)
    interp_video_frames = np.astype(insert_between_large_diffs(video_frames), np.int32)
    print(video_frames.shape, interp_video_frames.shape)

    fs = 176400
    dur = 3e-3
    hi_freq = 60e3
    low_freq = 20e3
    output_threshold = -50 # [dB]
    distance_threshold = 20 # [cm]

    METHOD = 'das' # 'das', 'capon'
    if METHOD == 'das':
        spatial_filter = das_filter
    elif METHOD == 'capon':
        spatial_filter = capon_method

    t_tone = np.linspace(0, dur, int(fs*dur))
    chirp = signal.chirp(t_tone, hi_freq, t_tone[-1], low_freq)    
    sig = pow_two_pad_and_window(chirp)

    C_AIR = 343
    min_distance = 10e-2
    discarded_samples = int(np.floor(((min_distance + 2.5e-2)*2)/C_AIR*fs))
    max_distance = 1
    max_index = int(np.floor(((max_distance + 2.5e-2)*2)/C_AIR*fs))

    def update(frame):
        # print(curr_end/fs)
        audio_data, _ = sf.read(robot_path, start=offsets[frame, 0], frames=offsets[frame, 1])
        # video_frame = int(offsets[frame, 0] / fs * video_fps) + start_frame
        # print('Video frame: %d' % video_frame)  
        dB_rms = 20*np.log10(np.mean(np.std(audio_data, axis=0)))    
        if dB_rms > output_threshold:
            filtered_signals = signal.correlate(audio_data, np.reshape(sig, (-1, 1)), 'same', method='fft')
            roll_filt_sigs = np.roll(filtered_signals, -len(sig)//2, axis=0)
            
            try:
                distance, direct_path, obst_echo = sonar(roll_filt_sigs, discarded_samples, max_index, fs)
                distance = distance*100 # [m] to [cm]
                theta, p = spatial_filter(
                                            roll_filt_sigs[obst_echo - int(5e-4*fs):obst_echo + int(5e-4*fs)], 
                                            fs=fs, nch=roll_filt_sigs.shape[1], d=2.70e-3, 
                                            bw=(low_freq, hi_freq)
                                        )
                p_dB = 10*np.log10(p)
                
                if direct_path != obst_echo:
                    doa_index = np.argmax(p_dB)
                    theta_hat = theta[doa_index]
                    if distance > 0:
                        # print('\nDistance: %.1f [cm] | DoA: %.2f [deg]' % (distance, theta_hat))            
                        return distance, theta_hat
                    else: return 0, 0
                else: return 0, 0
            except ValueError:
                print('\nNo valid distance or DoA')
                return 0, 0
        else:
            return 0, 0

except ffmpeg.Error as e:
    print('ffmpeg error:', e.stderr.decode(), file=sys.stderr)
    sys.exit(1)
try:
    frame_count = 0
    trajectory = np.zeros((0, 2), dtype=np.float32)
    counter = 0
    true_counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # print(interp_video_frames[counter], frame_count)
        # print(frame_count)
        if frame_count == interp_video_frames[counter]:
            counter += 1
            if frame_count in video_frames: 
                distance, doa = update(true_counter)
                true_counter += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect markers
            detector = aruco.ArucoDetector(aruco_dict, parameters)
            corners, ids, _ = detector.detectMarkers(gray)

            # print(f"Frame {frame_count}: Detected IDs: {ids}")
            
            # Draw detected markers
            if ids is not None:
                corners_array = np.squeeze(np.array(corners))
                aruco.drawDetectedMarkers(frame, corners, ids)
                try:
                    index = np.where(ids == robot_id)[0] # Find the index of the robot marker
                    # print(index)
                    arena_mrkr_indx = {
                        '12': np.where(ids == 12)[0], # Find the index of the arena markers
                        '13': np.where(ids == 13)[0], # Find the index of the arena markers
                        '14': np.where(ids == 14)[0], # Find the index of the arena markers
                    } # bottom left of 12, top left of 13, top right of 14
                    try: 
                        corners_12 = corners_array[arena_mrkr_indx['12']]
                        corners_13 = corners_array[arena_mrkr_indx['13']]
                        corners_14 = corners_array[arena_mrkr_indx['14']]
                        pixel_per_meters = np.mean([np.linalg.norm(corners_12[:, 3] - corners_13[:, 0], axis=1)/arena_w, np.linalg.norm(corners_13[:, 0] - corners_14[:, 1], axis=1)/arena_l])
                        center = np.mean(corners_array[index], axis=1)[0]
                        trajectory = np.append(trajectory, np.array([[center[0], center[1]]]), axis=0)
                        tl, tr, br, bl = np.squeeze(corners_array[index])
                        mic_positions = np.astype(get_offset_point(center, tl, tr, offset=-pixel_per_meters*0.055), np.int32)
                        
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
                                # print distance and angle                                
                                if distance != 0:
                                    print("Distance: %.1f [cm], Angle: %.1f [deg]" % (distance, doa))
                                    print("GT Distance: %.1f [cm], GT Angle: %.1f [deg]\n" % (sd, np.rad2deg(angle)))
                                cv2.line(frame, mic_positions, closest_obstacle.astype(int), (255, 51, 20), 2)
                                # print angle and id of nearest obstacle
                                # cv2.putText(frame, str(obst_ids[np.where(distances == sd)][0]), closest_obstacle.astype(int), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                # print(angle, frame_count)
                    
                        if len(trajectory) > 2:
                            # Draw trajectory
                            for i in range(len(trajectory) - 1):
                                cv2.line(frame, tuple(trajectory[i].astype(int)), tuple(trajectory[i + 1].astype(int)), (0, 255, 0), 2)
                        # Display result
                        resized_frame = cv2.resize(frame, (screen_width, screen_height))
                        cv2.imshow('ArUco Tracker', resized_frame)
                    except:
                        print("No arena marker detected")
                        pass
                except:
                    print("No robot marker detected")
                    pass
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print(frame_count, e)
    traceback.print_exc()
cap.release()
cv2.destroyAllWindows()
