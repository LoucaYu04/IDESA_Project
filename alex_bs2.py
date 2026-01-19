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
corner_ids = [4, 1, 2, 3]  # Replace with your actual corner marker IDs

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
            cv2.polylines(frame, [np.int32(ordered_corners)], isClosed=True, color=(0,255,0), thickness=2)
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
if corner_centers is not None:
    # --- Window size for warped area ---
    WARP_W, WARP_H = 600, 600
    dst_rect = np.array([[0,0],[WARP_W-1,0],[WARP_W-1,WARP_H-1],[0,WARP_H-1]], dtype=np.float32)
    ball_path_warped = []
    last_arc_center = None
    last_arc_radius = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break
        frame = gray_world_correction(frame)
        # Use the full camera frame for the tracking window (no perspective warp)
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
        # Always draw the fixed area (rectangle) for reference (green box)
        if corner_centers is not None:
            cv2.polylines(warped, [np.int32(corner_centers)], isClosed=True, color=(0,255,0), thickness=2)

        # Prepare data for Simulink (single target)
        inner_positions = {}
        if target_id in marker_centers:
            x, y = marker_centers[target_id]
            inner_positions[target_id] = (x / WARP_W, y / WARP_H)
            cv2.circle(warped, tuple(np.int32(marker_centers[target_id])), 8, (0,0,255), -1)
            cv2.putText(warped, f"({x/WARP_W:.2f}, {y/WARP_H:.2f})", tuple(np.int32(marker_centers[target_id]+10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # --- Red Ball Detection and Tracking ---
        ball_position = None
        if 'ball_path_warped' not in globals():
            ball_path_warped = []  # List of (point, timestamp)
        PATH_DURATION = 2.0  # seconds to keep in path
        # Only track if both ArUco markers are found
        if target_id in inner_positions:
            # --- Blue Line Detection and Arc Calculation ---
            hsv_img = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
            # Enlarged HSV range for blue
            lower_blue = np.array([85, 50, 30])   # H, S, V
            upper_blue = np.array([140, 255, 255])
            mask_blue = cv2.inRange(hsv_img, lower_blue, upper_blue)
            lines = cv2.HoughLinesP(mask_blue, 1, np.pi/180, threshold=15, minLineLength=20, maxLineGap=10)
            if lines is not None and len(lines) >= 2:
                # Find the two longest lines (assume these are the two parallel blue lines)
                sorted_lines = sorted(lines, key=lambda l: np.linalg.norm([l[0][2]-l[0][0], l[0][3]-l[0][1]]), reverse=True)
                line1 = sorted_lines[0][0]
                line2 = sorted_lines[1][0]
                # Draw both blue lines for visualization
                cv2.line(warped, (line1[0], line1[1]), (line1[2], line1[3]), (255,0,0), 3)
                cv2.line(warped, (line2[0], line2[1]), (line2[2], line2[3]), (255,0,0), 3)
                # Compute the midpoints of both lines
                mid1 = np.array([(line1[0] + line1[2]) / 2, (line1[1] + line1[3]) / 2])
                mid2 = np.array([(line2[0] + line2[2]) / 2, (line2[1] + line2[3]) / 2])
                # Use the midpoint between the two lines as the 'ball' position
                x_ball, y_ball = ((mid1 + mid2) / 2)
                center_ball = (int(x_ball), int(y_ball))
                ball_position = center_ball
                # Use the average direction of the two lines as the blue direction
                dir1 = np.array([line1[2] - line1[0], line1[3] - line1[1]], dtype=np.float32)
                dir2 = np.array([line2[2] - line2[0], line2[3] - line2[1]], dtype=np.float32)
                blue_dir = (dir1 + dir2) / 2
                if np.linalg.norm(blue_dir) > 1e-3:
                    blue_dir = blue_dir / np.linalg.norm(blue_dir)
            elif lines is not None and len(lines) == 1:
                # Fallback: only one line detected, use as before
                xA, yA, xB, yB = lines[0][0]
                cv2.line(warped, (xA, yA), (xB, yB), (255,0,0), 3)
                x_ball = (xA + xB) / 2
                y_ball = (yA + yB) / 2
                center_ball = (int(x_ball), int(y_ball))
                ball_position = center_ball
                blue_dir = np.array([xB-xA, yB-yA], dtype=np.float32)
                if np.linalg.norm(blue_dir) > 1e-3:
                    blue_dir = blue_dir / np.linalg.norm(blue_dir)
                # --- Ball path tracking ---
                now = time.time()
                ball_path_warped.append((center_ball, now))
                # Remove points older than PATH_DURATION seconds
                ball_path_warped = [(pt, t) for pt, t in ball_path_warped if now - t <= PATH_DURATION]
                # Draw the path (only recent points)
                if len(ball_path_warped) > 1:
                    pts = np.array([pt for pt, t in ball_path_warped], dtype=np.int32)
                    cv2.polylines(warped, [pts], False, (0,0,255), 2)
                # Draw major direction arrow (from recent path)
                major_window = min(20, len(ball_path_warped)-1)
                if major_window > 2:
                    pt1 = ball_path_warped[-major_window][0]
                    pt2 = ball_path_warped[-1][0]
                    major_vec = np.array(pt2) - np.array(pt1)
                    if np.linalg.norm(major_vec) > 10:
                        cv2.arrowedLine(warped, tuple(pt1), tuple(pt2), (0,255,255), 3, tipLength=0.3)
                # Draw line to the target marker only
                if target_id in marker_centers:
                    cv2.line(warped, center_ball, tuple(np.int32(marker_centers[target_id])), (0,0,255), 2)
                    # Map ball position to normalized coordinates
                    x_norm, y_norm = x_ball / WARP_W, y_ball / WARP_H
                    cv2.putText(warped, f"Ball ({x_norm:.2f}, {y_norm:.2f})", (center_ball[0]+10, center_ball[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    # --- Arc calculation: draw only the arc segment from blue line to each marker ---
                    ball_dir = blue_dir
                    ball_pos = np.array([x_ball, y_ball])
                    if target_id in marker_centers:
                        target_pos = np.array(marker_centers[target_id])
                        v = ball_dir / np.linalg.norm(ball_dir)
                        w = target_pos - ball_pos
                        # Project w onto v to get the tangent point
                        proj = np.dot(w, v) * v
                        perp = w - proj
                        if np.linalg.norm(perp) < 1e-6:
                            # If the marker is directly in line, just draw a straight line
                            cv2.line(warped, tuple(center_ball), tuple(np.int32(target_pos)), (0,255,255), 3)
                        else:
                            # Find the center of the circle passing through ball_pos and target_pos, with tangent v at ball_pos
                            # The center lies at ball_pos + normal * r, where r = |w|^2 / (2 * |perp|)
                            r = np.dot(w, w) / (2 * np.linalg.norm(perp))
                            normal = np.array([-v[1], v[0]])
                            # Choose the normal direction so that the center is on the same side as perp
                            if np.dot(perp, normal) < 0:
                                normal = -normal
                            center = ball_pos + normal * r
                            radius = np.linalg.norm(center - ball_pos)
                            # Draw the arc from ball_position to target_pos
                            def draw_arc_segment(img, center, radius, pt1, pt2, color, thickness=2):
                                angle = lambda c, p: math.atan2(p[1]-c[1], p[0]-c[0])
                                a1 = angle(center, pt1)
                                a2 = angle(center, pt2)
                                # Ensure shortest arc direction
                                delta = (a2 - a1) % (2*math.pi)
                                if delta > math.pi:
                                    a1, a2 = a2, a1
                                    delta = (a2 - a1) % (2*math.pi)
                                num_pts = max(10, int(abs(delta)*radius/5))
                                arc_pts = [
                                    (
                                        int(center[0] + radius * math.cos(a1 + t * delta / num_pts)),
                                        int(center[1] + radius * math.sin(a1 + t * delta / num_pts))
                                    )
                                    for t in range(num_pts+1)
                                ]
                                cv2.polylines(img, [np.array(arc_pts, dtype=np.int32)], False, color, thickness)

                            if ball_position is not None:
                                draw_arc_segment(
                                    warped,
                                    center,
                                    radius,
                                    center_ball,
                                    marker_centers[target_id],
                                    (0,255,255),
                                    3
                                )
                    # Send ball position, arc length, and radius to Simulink
                    # Removed string UDP send to Simulink; only double array is sent now
                    # --- Calculate speed and tilt angle for Simulink ---
                    if ball_position is not None and target_id in marker_centers:
                        target_pos = np.array(marker_centers[target_id])
                        ball_pos = np.array(ball_position)
                        # Distance to target (normalized and pixel)
                        distance = np.linalg.norm(target_pos - ball_pos) / WARP_W
                        pixel_distance = np.linalg.norm(target_pos - ball_pos)
                        catchment_radius = 75
                        slow_radius = 240
                        # Ball velocity vector (from path)
                        if len(ball_path_warped) > 2:
                            prev_ball_pos = np.array(ball_path_warped[-2][0])
                            velocity_vec = ball_pos - prev_ball_pos
                        else:
                            velocity_vec = target_pos - ball_pos  # fallback: point toward target
                        # Always move toward the target (speed always positive toward target)
                        if pixel_distance <= catchment_radius:
                            while True:
                                speed = 0
                                tilt_angle = 0
                                print(f"Ball reached catchment area of ArUco {target_id}. Stopping.\n")
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
                                    break
                                except ValueError:
                                    print("Invalid input. Please enter an integer.")
                        else:
                            # Speed ramps up if far, slows down if close
                            speed = 1 * (2 / (1 + np.exp(-8 * (distance - 0.2))) - 1)
                            speed = abs(speed)
                            if pixel_distance <= slow_radius:
                                speed = 1 * (pixel_distance - catchment_radius) / (slow_radius - catchment_radius)
                                speed = np.clip(speed, 0, 1)
                            speed = np.clip(speed, 0, 1)

                        # Tilt angle: steer based on angle off the direct line to target
                        direction_vec = target_pos - ball_pos
                        direction_vec_norm = direction_vec / (np.linalg.norm(direction_vec) + 1e-6)
                        velocity_norm = velocity_vec / (np.linalg.norm(velocity_vec) + 1e-6)
                        # Angle between velocity and direct line to target
                        angle_off = np.arcsin(np.cross(direction_vec_norm, velocity_norm))
                        tilt_angle = -5 * angle_off  # Negative to steer toward the line

                        # Throttle UDP sending to 100 times per second (every 0.01 seconds)
                        if 'last_udp_send_time' not in globals():
                            last_udp_send_time = 0
                        now = time.time()
                        if now - last_udp_send_time >= 0.01:
                            udp_array = np.array([speed, tilt_angle], dtype=np.float64)
                            sock_send.sendto(udp_array.tobytes(), (UDP_IP_SEND, UDP_PORT))
                            print(f"[UDP] Sent speed: {speed:.2f}, tilt_angle: {tilt_angle:.2f}")
                            last_udp_send_time = now
        else:
            cv2.polylines(warped, [np.int32(dst_rect)], isClosed=True, color=(0,255,0), thickness=2)
        cv2.imshow("Live Tracking", warped)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()