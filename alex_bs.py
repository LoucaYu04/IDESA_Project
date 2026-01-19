import cv2
import cv2.aruco as aruco
import numpy as np
import time
import socket
import math
import time


# Setup UDP communication parameters
UDP_IP_SEND = "138.38.227.25"
UDP_IP_RECEIVE = "172.26.109.96"  # LOUCA'S LAPTOP IP
UDP_PORT = 25000

# Create separate sockets for sending and receiving
sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
print("Listening on IP:", UDP_IP_RECEIVE, "Port:", UDP_PORT)


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




# --- Main Live Tracking Loop ---
WARP_W, WARP_H = 600, 600
ball_path_warped = []
last_arc_center = None
last_arc_radius = None
PATH_DURATION = 2.0  # seconds to keep in path
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
    warped = frame.copy()
    # Detect ArUco markers in the full frame for position mapping
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

    # Prepare data for Simulink (single target)
    inner_positions = {}
    if target_id in marker_centers:
        x, y = marker_centers[target_id]
        inner_positions[target_id] = (x / WARP_W, y / WARP_H)
        cv2.circle(warped, tuple(np.int32(marker_centers[target_id])), 8, (0,0,255), -1)
        cv2.putText(warped, f"({x/WARP_W:.2f}, {y/WARP_H:.2f})", tuple(np.int32(marker_centers[target_id]+10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # --- Red Ball Detection and Tracking ---
        # --- Ball Detection: From Two Parallel Blue Lines ---
        ball_position = None
        x_ball, y_ball = None, None
        center_ball = None
        # Convert to HSV and mask for blue (wider range, no orange decontamination)
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        # Widened blue range: covers more blue shades
        lower_blue = np.array([90, 80, 40])   # H, S, V
        upper_blue = np.array([140, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        # Morph open to clean up
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(mask, 1, np.pi/180, threshold=60, minLineLength=80, maxLineGap=30)
        blue_lines = []
        if lines is not None:
            for l in lines:
                x1, y1, x2, y2 = l[0]
                # Only keep lines that are roughly vertical (for parallel check)
                angle = np.arctan2(y2-y1, x2-x1)
                angle_deg = np.degrees(angle)
                if abs(angle_deg) > 70:
                    blue_lines.append(((x1, y1), (x2, y2)))
                    cv2.line(warped, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # Find two longest, most parallel lines
        best_pair = None
        max_length = 0
        if len(blue_lines) >= 2:
            for i in range(len(blue_lines)):
                for j in range(i+1, len(blue_lines)):
                    (a1, a2) = blue_lines[i]
                    (b1, b2) = blue_lines[j]
                    # Compute direction vectors
                    va = np.array([a2[0]-a1[0], a2[1]-a1[1]])
                    vb = np.array([b2[0]-b1[0], b2[1]-b1[1]])
                    va = va / (np.linalg.norm(va)+1e-6)
                    vb = vb / (np.linalg.norm(vb)+1e-6)
                    dot = np.dot(va, vb)
                    # Check if nearly parallel
                    if dot > 0.98:
                        # Use sum of lengths as score
                        la = np.linalg.norm(np.array(a2)-np.array(a1))
                        lb = np.linalg.norm(np.array(b2)-np.array(b1))
                        if la+lb > max_length:
                            max_length = la+lb
                            best_pair = (blue_lines[i], blue_lines[j])
        if best_pair is not None:
            (l1, l2) = best_pair
            # Midpoint of each line
            m1 = ((l1[0][0]+l1[1][0])//2, (l1[0][1]+l1[1][1])//2)
            m2 = ((l2[0][0]+l2[1][0])//2, (l2[0][1]+l2[1][1])//2)
            # Ball position is midpoint between the two midpoints
            ball_position = ((m1[0]+m2[0])//2, (m1[1]+m2[1])//2)
            x_ball, y_ball = ball_position
            center_ball = ball_position
            cv2.circle(warped, ball_position, 12, (0,255,255), 2)
            cv2.circle(warped, ball_position, 3, (0,0,255), -1)
            # Draw a line between the two midpoints for visualization
            cv2.line(warped, m1, m2, (0,255,255), 2)
        # Ball path tracking
        now = time.time()
        if ball_position is not None:
            ball_path_warped.append((ball_position, now))
            ball_path_warped = [(pt, t) for pt, t in ball_path_warped if now - t <= PATH_DURATION]
        # Draw the path (only recent points)
        if len(ball_path_warped) > 1:
            pts = np.array([pt for pt, t in ball_path_warped], dtype=np.int32)
            cv2.polylines(warped, [pts], False, (0,0,255), 2)
        # Draw line from ball to target marker
        if ball_position is not None and target_id in marker_centers:
            target_xy = tuple(np.int32(marker_centers[target_id]))
            cv2.line(warped, ball_position, target_xy, (0,255,255), 2)
            # Map ball position to normalized coordinates
            x_norm, y_norm = x_ball / WARP_W, y_ball / WARP_H
            cv2.putText(warped, f"Ball ({x_norm:.2f}, {y_norm:.2f})", (center_ball[0]+10, center_ball[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # --- Control logic: move toward target, steer if moving away ---
        if ball_position is not None and target_id in marker_centers:
            target_pos = np.array(marker_centers[target_id])
            ball_pos = np.array(ball_position)
            # Distance to target (normalized and pixel)
            distance = np.linalg.norm(target_pos - ball_pos) / WARP_W
            pixel_distance = np.linalg.norm(target_pos - ball_pos)
            catchment_radius = 75
            slow_radius = 240
            # Ball movement direction (velocity vector)
            if len(ball_path_warped) > 2:
                prev_ball_pos = np.array(ball_path_warped[-2][0])
                movement_vec = ball_pos - prev_ball_pos
            else:
                movement_vec = np.array([0,0])
            # Direction to target
            direction_vec = target_pos - ball_pos
            direction_vec_norm = direction_vec / (np.linalg.norm(direction_vec) + 1e-6)
            # If ball is moving away from the target, steer more toward the line
            moving_away = np.dot(movement_vec, direction_vec_norm) < -2  # threshold for 'moving away'
            # Speed logic
            if pixel_distance <= catchment_radius:
                while True:
                    speed = 0
                    print(f"Ball reached catchment area of ArUco {target_id}. Stopping.\n")
                    tilt_angle = 0
                    udp_array = np.array([speed, tilt_angle], dtype=np.float64)
                    sock_send.sendto(udp_array.tobytes(), (UDP_IP_SEND, UDP_PORT))
                    print(f"[UDP] Sent speed: {speed:.2f}, tilt_angle: {tilt_angle:.2f}")
                    try:
                        new_id = int(input("Enter the ArUco ID of the next target marker: "))
                        if new_id == target_id:
                            print("New target ID must be different from the previous one.")
                            continue
                        target_id = new_id
                        print(f"Now tracking ArUco marker {target_id}.")
                        if target_id in marker_centers and ball_position is not None:
                            target_pos = np.array(marker_centers[target_id])
                            ball_pos = np.array(ball_position)
                            pixel_distance = np.linalg.norm(target_pos - ball_pos)
                            if pixel_distance <= catchment_radius:
                                continue
                    except ValueError:
                        print("Invalid input. Please enter an integer.")
                    break
            else:
                # Speed ramps up if far, slows down if close
                speed = 1 * (2 / (1 + np.exp(-8 * (distance - 0.2))) - 1)
                speed = abs(speed)
                if pixel_distance <= slow_radius:
                    speed = 1 * (pixel_distance - catchment_radius) / (slow_radius - catchment_radius)
                    speed = np.clip(speed, 0, 1)
                speed = np.clip(speed, 0, 1)
            # Tilt angle: steer toward the target line, or correct if moving away
            angle_to_target = math.atan2(direction_vec[1], direction_vec[0])
            if moving_away:
                # If moving away, steer more aggressively
                tilt_angle = 23 * np.sign(np.cross(movement_vec, direction_vec))
            else:
                tilt_angle = 23 * np.sin(angle_to_target)
            # Clamp tilt angle
            tilt_angle = np.clip(tilt_angle, -23, 23)
            # Throttle UDP sending to 100 times per second (every 0.01 seconds)
            if 'last_udp_send_time' not in globals():
                last_udp_send_time = 0
            now = time.time()
            if now - last_udp_send_time >= 0.01:
                udp_array = np.array([speed, tilt_angle], dtype=np.float64)
                sock_send.sendto(udp_array.tobytes(), (UDP_IP_SEND, UDP_PORT))
                print(f"[UDP] Sent speed: {speed:.2f}, tilt_angle: {tilt_angle:.2f}")
                last_udp_send_time = now
    cv2.imshow("Live Tracking", warped)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()