import cv2
import cv2.aruco as aruco
import numpy as np
import time
import socket
import math
import time

#Setup UDP communication parameters
UDP_IP_SEND = "138.38.228.99"
UDP_IP_RECEIVE = "172.26.109.96" #LOUCA'S LAPTOP IP
UDP_PORT = 25000

""" # Create separate sockets for sending and receiving
sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_recv.bind((UDP_IP_RECEIVE, UDP_PORT))
sock_recv.setblocking(False)  # Non-blocking mode
print("Listening on IP:", UDP_IP_RECEIVE, "Port:", UDP_PORT) """


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
        # Perspective transform to crop to the area
        H_warp, _ = cv2.findHomography(corner_centers, dst_rect)
        warped = cv2.warpPerspective(frame, H_warp, (WARP_W, WARP_H))
        # Detect ArUco markers in the original frame for position mapping
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        marker_centers = {}
        if ids is not None:
            ids_flat = ids.flatten()
            aruco.drawDetectedMarkers(warped, corners, ids)
            for i, marker_id in enumerate(ids_flat):
                c = corners[i][0]
                center = np.mean(c, axis=0)
                # Map marker center to warped image
                pt = np.array([*center, 1.0])
                mapped = H_warp @ pt
                mapped /= mapped[2]
                marker_centers[marker_id] = mapped[:2]
            # Draw the fixed area (rectangle)
            cv2.polylines(warped, [np.int32(dst_rect)], isClosed=True, color=(0,255,0), thickness=2)

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
                lower_blue = np.array([100, 120, 70])
                upper_blue = np.array([130, 255, 255])
                mask_blue = cv2.inRange(hsv_img, lower_blue, upper_blue)
                lines = cv2.HoughLinesP(mask_blue, 1, np.pi/180, threshold=20, minLineLength=40, maxLineGap=15)
                if lines is not None:
                    # Pick the longest line as the direction
                    longest = max(lines, key=lambda l: np.linalg.norm([l[0][2]-l[0][0], l[0][3]-l[0][1]]))
                    xA, yA, xB, yB = longest[0]
                    # Draw the blue line on the main image for visualization
                    cv2.line(warped, (xA, yA), (xB, yB), (255,0,0), 3)
                    # Use the midpoint as the 'ball' position
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
                    msg = f"ball,{x_norm:.4f},{y_norm:.4f};arc_radius,{last_arc_radius if last_arc_radius is not None else 0:.4f};" + \
                        f"{target_id},{inner_positions[target_id][0]:.4f},{inner_positions[target_id][1]:.4f}" if target_id in inner_positions else ""
                    # sock_send.sendto(msg.encode(), (UDP_IP_SEND, UDP_PORT))
                    # --- Calculate distance and arc angle to target ArUco marker ---
                    if last_arc_center is not None and last_arc_radius is not None and ball_position is not None and target_id in marker_centers:
                        # Distance from ball to target marker
                        target_pos = np.array(marker_centers[target_id])
                        ball_pos = np.array(ball_position)
                        distance = np.linalg.norm(target_pos - ball_pos)
                        # Convert distance to steps (u
                        # 
                        # ser input required)
                        MOTOR_STEP_MM = 5  # Example: 5mm per step (change as needed)
                        steps = int(distance / MOTOR_STEP_MM)
                        # Arc angle in degrees
                        if last_arc_center is not None:
                            def angle_between(center, pt):
                                return math.atan2(pt[1]-center[1], pt[0]-center[0])
                            angle1 = angle_between(last_arc_center, ball_pos)
                            angle2 = angle_between(last_arc_center, target_pos)
                            arc_angle_rad = abs(angle2 - angle1)
                            arc_angle_rad = min(arc_angle_rad, 2*math.pi - arc_angle_rad)
                            arc_angle_deg = int(np.degrees(arc_angle_rad))
                        else:
                            arc_angle_deg = 0
                        # Prepare int8 array for UDP
                        steps_int8 = np.clip(steps, -128, 127)
                        angle_int8 = np.clip(arc_angle_deg, -128, 127)
                        udp_array = np.array([steps_int8, angle_int8], dtype=np.int8)
                        #sock_send.sendto(udp_array.tobytes(), (UDP_IP_SEND, UDP_PORT))
                        print(f"[UDP] Sent steps: {steps_int8}, angle: {angle_int8}")
        else:
            cv2.polylines(warped, [np.int32(dst_rect)], isClosed=True, color=(0,255,0), thickness=2)
        cv2.imshow("Live Tracking", warped)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()