import cv2
import cv2.aruco as aruco
import numpy as np
import time
import socket
import math
import time

# --- Tkinter for GUI ---
import threading
import tkinter as tk


# Setup UDP communication parameters
UDP_IP_SEND = "138.38.228.97"
UDP_IP_RECEIVE = "172.26.109.96"  # LOUCA'S LAPTOP IP
UDP_PORT = 25000

# Create separate sockets for sending and receiving
sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
print("Listening on IP:", UDP_IP_RECEIVE, "Port:", UDP_PORT)

# --- UDP socket for color signals (reuse sock_send) ---
def send_udp_color_signal(color_id):
    msg = bytes([color_id])
    sock_send.sendto(msg, (UDP_IP_SEND, UDP_PORT))
    print(f"[UDP] Sent color signal: {color_id}")


                    #------CAMERA CALIBRATION------
# Load the camera calibration values
camera_calibration = np.load('Calibration1.npz')
CM=camera_calibration['CM'] #camera matrix
dist_coef=camera_calibration['dist_coef']# distortion coefficients from the camera

# Define the ArUco dictionary and parameters
marker_size = 95
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

# Define a processing rate
processing_period = 0.25

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# Attempt to set white balance to auto (if supported)
if hasattr(cv2, 'CAP_PROP_AUTO_WB'):
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)
if hasattr(cv2, 'CAP_PROP_WB_TEMPERATURE'):
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4500)  # 4500K is a neutral value

# Set the starting time
start_time = time.time()
fps = 0


# --- Functions to run for each color ---
def on_red_detected():
    print("Red detected! Running red function...")

def on_green_detected():
    print("Green detected! Running green function...")

def on_blue_detected():
    print("Blue detected! Running blue function...")

# --- ArUco Area Tracking Setup ---


# Set your ArUco marker IDs here
corner_ids = [0, 1, 2, 3]  # Replace with your actual corner marker IDs

# Prompt user for the target ArUco marker ID
while True:
    try:
        target_id = int(input("Enter the ArUco ID of the target marker: "))
        break
    except ValueError:
        print("Invalid input. Please enter an integer.")

def order_corners(corners_dict):
    # Returns corners in order: [top-left, top-right, bottom-right, bottom-left]
    pts = np.array([corners_dict[i] for i in corner_ids])
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    ordered = np.zeros((4,2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]      # top-left
    ordered[2] = pts[np.argmax(s)]      # bottom-right
    ordered[1] = pts[np.argmin(diff)]   # top-right
    ordered[3] = pts[np.argmax(diff)]   # bottom-left
    return ordered

aruco_path = []
prev_color = None


# --- PHASE 1: Set the 4 corner ArUco codes ---
corner_centers = None
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from camera.")
        break

    def gray_world_correction(img):
        img = img.astype(np.float32)
        avg_b = np.mean(img[:,:,0])
        avg_g = np.mean(img[:,:,1])
        avg_r = np.mean(img[:,:,2])
        avg_gray = (avg_b + avg_g + avg_r) / 3
        img[:,:,0] = np.clip(img[:,:,0] * (avg_gray / avg_b), 0, 255)
        img[:,:,1] = np.clip(img[:,:,1] * (avg_gray / avg_g), 0, 255)
        img[:,:,2] = np.clip(img[:,:,2] * (avg_gray / avg_r), 0, 255)
        return img.astype(np.uint8)

    frame = gray_world_correction(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    marker_centers = {}
    if ids is not None:
        ids_flat = [int(i) for i in ids.flatten()]
        print(f"Detected marker IDs: {ids_flat}")
        aruco.drawDetectedMarkers(frame, corners, ids)
        for i, marker_id in enumerate(ids_flat):
            c = corners[i][0]
            center = np.mean(c, axis=0)
            marker_centers[marker_id] = center
        # Ensure corner_ids are also int for comparison
        if all(int(cid) in marker_centers for cid in corner_ids):
            ordered_corners = order_corners(marker_centers)
            cv2.polylines(frame, [np.int32(ordered_corners)], isClosed=True, color=(255,0,0), thickness=2)
            corner_centers = ordered_corners.copy()
            cv2.putText(frame, "Corners set! Press 'q' to continue", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        else:
            cv2.putText(frame, "Show all 4 corner ArUco codes", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    else:
        cv2.putText(frame, "Show all 4 corner ArUco codes", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Set Corners", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- PHASE 2: Live tracking with fixed corners ---

# --- Combined Tkinter GUI and OpenCV feed ---
if corner_centers is not None:
    WARP_W, WARP_H = 600, 600
    dst_rect = np.array([[0,0],[WARP_W-1,0],[WARP_W-1,WARP_H-1],[0,WARP_H-1]], dtype=np.float32)
    ball_path_warped = []
    last_arc_center = None
    last_arc_radius = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = gray_world_correction(frame)
        warped = frame.copy()
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        marker_centers = {}
        if ids is not None:
            ids_flat = ids.flatten()
            aruco.drawDetectedMarkers(warped, corners, ids)
            for i, marker_id in enumerate(ids_flat):
                c = corners[i][0]
                center = np.mean(c, axis=0)
                marker_centers[marker_id] = center
        if corner_centers is not None:
            cv2.polylines(warped, [np.int32(corner_centers)], isClosed=True, color=(255,0,0), thickness=2)

        inner_positions = {}
        if target_id in marker_centers:
            x, y = marker_centers[target_id]
            inner_positions[target_id] = (x / WARP_W, y / WARP_H)
            cv2.circle(warped, tuple(np.int32(marker_centers[target_id])), 8, (0,0,255), -1)
            cv2.putText(warped, f"({x/WARP_W:.2f}, {y/WARP_H:.2f})", tuple(np.int32(marker_centers[target_id]+10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        ball_position = None
        if 'ball_path_warped' not in globals():
            ball_path_warped = []
        PATH_DURATION = 2.0

        if target_id in inner_positions:
            hsv_img = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
            # Yellow line detection (replace green)
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([35, 255, 255])
            mask_yellow = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
            lines = cv2.HoughLinesP(mask_yellow, 1, np.pi/180, threshold=20, minLineLength=20, maxLineGap=15)
            if lines is not None:
                longest = max(lines, key=lambda l: np.linalg.norm([l[0][2]-l[0][0], l[0][3]-l[0][1]]))
                xA, yA, xB, yB = longest[0]
                cv2.line(warped, (xA, yA), (xB, yB), (0,255,255), 3)  # Draw yellow line
                x_ball = (xA + xB) / 2
                y_ball = (yA + yB) / 2
                center_ball = (int(x_ball), int(y_ball))
                ball_position = center_ball
                if target_id in marker_centers:
                    target_xy = tuple(np.int32(marker_centers[target_id]))
                    cv2.line(warped, center_ball, target_xy, (0,0,255), 2)
                    x_norm, y_norm = x_ball / WARP_W, y_ball / WARP_H
                    cv2.putText(warped, f"Ball ({x_norm:.2f}, {y_norm:.2f})", (center_ball[0]+10, center_ball[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

                now = time.time()
                if 'ball_path_warped' not in locals():
                    ball_path_warped = []
                ball_path_warped.append((center_ball, now))
                ball_path_warped = [(pt, t) for pt, t in ball_path_warped if now - t <= 2.0]
                if target_id in marker_centers:
                    target_pos = np.array(marker_centers[target_id])
                    ball_pos = np.array(center_ball)
                    direction_vec = target_pos - ball_pos
                    distance = np.linalg.norm(direction_vec)
                    direction_vec_norm = direction_vec / (distance + 1e-6)
                    if len(ball_path_warped) >= 2:
                        prev_ball_pos = np.array(ball_path_warped[-2][0])
                        velocity_vec = ball_pos - prev_ball_pos
                        velocity_norm = velocity_vec / (np.linalg.norm(velocity_vec) + 1e-6)
                        catchment_radius = 50
                        slow_radius = 240
                        if distance < catchment_radius:
                            speed = 0.0
                        else:
                            speed = 1 * (2 / (1 + np.exp(-8 * (distance/WARP_W - 0.2))) - 1)
                            speed = abs(speed)
                            if distance <= slow_radius:
                                speed = np.clip(speed, 0, 0.75)
                            speed = np.clip(speed, 0, 1)
                            if speed > 0.2 and speed < 0.8:
                                speed = 0.8
                            moving_away = np.dot(velocity_norm, direction_vec_norm) < 0
                            #if moving_away:
                            #    speed = -speed
                            # Stable signed angle using atan2
                            angle_off = np.arctan2(
                                direction_vec_norm[0]*velocity_norm[1] - direction_vec_norm[1]*velocity_norm[0],
                                direction_vec_norm[0]*velocity_norm[0] + direction_vec_norm[1]*velocity_norm[1]
                            )
                            tilt_angle = 15 * angle_off
                    else:
                        speed = 0.0
                        tilt_angle = 0.0

                    # --- Draw arrows for speed and tilt angle on the ball ---
                    if center_ball is not None:
                        # Arrow for speed (direction_vec_norm)
                        max_arrow_len = 80  # pixels for speed=1
                        speed_arrow_len = int(max_arrow_len * abs(speed))
                        # Forward direction (direction_vec_norm)
                        dir_vec = direction_vec_norm if speed >= 0 else -direction_vec_norm
                        end_speed = (int(center_ball[0] + dir_vec[0] * speed_arrow_len), int(center_ball[1] + dir_vec[1] * speed_arrow_len))
                        cv2.arrowedLine(warped, center_ball, end_speed, (0, 255, 0), 4, tipLength=0.3)
                        cv2.putText(warped, f"Speed: {speed:.2f}", (center_ball[0] + 10, center_ball[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                        # Arrow for tilt angle (perpendicular to direction_vec_norm)
                        max_tilt = 15.0
                        tilt_scale = min(abs(tilt_angle) / max_tilt, 1.0)
                        tilt_arrow_len = int(max_arrow_len * tilt_scale)
                        # Perpendicular direction (right for positive tilt, left for negative)
                        perp_vec = np.array([-dir_vec[1], dir_vec[0]])
                        if tilt_angle < 0:
                            perp_vec = -perp_vec
                        end_tilt = (int(center_ball[0] + perp_vec[0] * tilt_arrow_len), int(center_ball[1] + perp_vec[1] * tilt_arrow_len))
                        cv2.arrowedLine(warped, center_ball, end_tilt, (0, 0, 255), 4, tipLength=0.3)
                        cv2.putText(warped, f"Tilt: {tilt_angle:.1f}", (center_ball[0] + 10, center_ball[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    if 'last_udp_send_time' not in globals():
                        last_udp_send_time = 0
                    now = time.time()
                    if now - last_udp_send_time >= 0.01:
                        udp_array = np.array([speed, tilt_angle], dtype=np.float64)
                        sock_send.sendto(udp_array.tobytes(), (UDP_IP_SEND, UDP_PORT))
                        print(f"[UDP] Sent speed: {speed:.2f}, tilt_angle: {tilt_angle:.2f}")
                        last_udp_send_time = now
        # Show the camera window
        cv2.imshow("Live Tracking", warped)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()