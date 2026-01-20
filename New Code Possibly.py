import cv2

import numpy as np
import time
import socket
import math

# --- Track points (user clicks) ---
track_points = []  # List of (x, y) tuples
current_target_idx = 0
def mouse_callback(event, x, y, flags, param):
    global track_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(track_points) < 5:
            track_points.append((x, y))
            print(f"Added track point: ({x}, {y})")
        else:
            print("Maximum of 5 points reached. Further clicks are ignored.")


# --- PHASE 1: Click points on the first window ---
cv2.namedWindow("Set Path Points", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Set Path Points", mouse_callback)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from camera.")
        break
    # Draw crosses and numbers for all clicked points
    for idx, pt in enumerate(track_points):
        x, y = pt
        color = (255, 255, 255) if idx < 4 else (0, 0, 0)  # 5th point is black
        cv2.drawMarker(frame, (int(x), int(y)), color, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
        cv2.putText(frame, str(idx+1), (int(x)+8, int(y)-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    cv2.imshow("Set Path Points", frame)
    # Allow user to press 'q' to continue only after 4 or 5 points are set
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') and len(track_points) >= 4:
        break
cv2.destroyWindow("Set Path Points")

# --- PHASE 2: Main tracking window ---
cv2.namedWindow("Live Tracking", cv2.WINDOW_NORMAL)

# Setup UDP communication parameters
UDP_IP_SEND = "138.38.228.97"
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



# Define a processing rate
processing_period = 0.25

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

# cap already opened above
# Attempt to set white balance to auto (if supported)
if hasattr(cv2, 'CAP_PROP_AUTO_WB'):
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)
if hasattr(cv2, 'CAP_PROP_WB_TEMPERATURE'):
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4500)  # 4500K is a neutral value

# Set the starting time
start_time = time.time()
fps = 0


# --- Functions to run for each color ---

# --- Color tracking in top-left region ---
def draw_arc_segment(img, center, radius, pt1, pt2, color, thickness=2):
    """Draw an arc from pt1 to pt2 around center with given radius."""
    def angle(c, p):
        return math.atan2(p[1]-c[1], p[0]-c[0])
    a1 = angle(center, pt1)
    a2 = angle(center, pt2)
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
def detect_color_in_region(img):
    # Only look at the top-left 1/6th of the image
    h, w = img.shape[:2]
    region = img[0:h//15, 0:w//15]
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    # Red mask (handle wraparound)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    # Green mask
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([80, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    # Blue mask
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    # Count pixels
    red_count = np.count_nonzero(mask_red)
    green_count = np.count_nonzero(mask_green)
    blue_count = np.count_nonzero(mask_blue)
    # Threshold for detection (tune as needed)
    threshold = 100
    if green_count > threshold:
        return 1  # Green
    elif red_count > threshold:
        return 2  # Red
    elif blue_count > threshold:
        return 3  # Blue
    else:
        return 0  # None

warped = frame.copy()
hsv_img = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([35, 255, 255])
mask_yellow = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
lines = cv2.HoughLinesP(mask_yellow, 1, np.pi/180, threshold=20, minLineLength=20, maxLineGap=15)
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

gray_world_correction = lambda img: img.astype(np.uint8)  # Dummy, since it was only used in ArUco phase
if True:
    # --- Window size for warped area ---
    WARP_W, WARP_H = 600, 600
    dst_rect = np.array([[0,0],[WARP_W-1,0],[WARP_W-1,WARP_H-1],[0,WARP_H-1]], dtype=np.float32)
    ball_path_warped = []
    last_arc_center = None
    last_arc_radius = None
    color_num = 0
    tilt_angle = 0  # Ensure tilt_angle is always defined
    speed = 0      # Ensure speed is always defined
    current_target_idx = 0  # Always start at the first point
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break
        frame = gray_world_correction(frame)
        # --- Color tracking in top-left region ---
        color_num = detect_color_in_region(frame)
        # Use the full camera frame for the tracking window (no perspective warp)
        warped = frame.copy()  # Ensure 'warped' is always defined before any use
        # --- Yellow line detection and ball_position update (must be after 'warped' is defined) ---
        hsv_img = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        mask_yellow = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
        lines = cv2.HoughLinesP(mask_yellow, 1, np.pi/180, threshold=20, minLineLength=20, maxLineGap=15)
        ball_position = None
        blue_dir = np.array([1, 0], dtype=np.float32)  # Default direction if no line
        if lines is not None:
            # Choose the line closest to the previous ball position, or the longest if no previous
            chosen_line = None
            if 'prev_ball_pos' in locals() and prev_ball_pos is not None:
                prev_pos = np.array(prev_ball_pos)
                def line_center(line):
                    xA, yA, xB, yB = line[0]
                    return np.array([(xA + xB) / 2, (yA + yB) / 2])
                chosen_line = min(lines, key=lambda l: np.linalg.norm(line_center(l) - prev_pos))
            else:
                # Fallback: pick the longest line
                chosen_line = max(lines, key=lambda l: np.linalg.norm([l[0][2]-l[0][0], l[0][3]-l[0][1]]))
            xA, yA, xB, yB = chosen_line[0]
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

        # Draw smaller crosses and numbers for all clicked points
        for idx, pt in enumerate(track_points):
            x, y = pt
            color = (255, 255, 255) if idx < 4 else (0, 0, 0)  # 5th point is black
            cv2.drawMarker(warped, (int(x), int(y)), color, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
            cv2.putText(warped, str(idx+1), (int(x)+8, int(y)-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        # --- Ball Detection and Tracking ---
        # ball_position will be set by yellow line detection below
        if 'ball_path_warped' not in globals():
            ball_path_warped = []  # List of (point, timestamp)
        PATH_DURATION = 2.0  # seconds to keep in path
        reversing_mode = False
        prev_distance = None
        prev_ball_pos = None
                    # --- Ball targeting logic using track points and arc ---
        # Main path cycling: only use first 4 points
        main_path_points = track_points[:4]
        use_fifth = len(track_points) == 5 and color_num == 3
        # Only proceed if yellow line is detected and ball_position is set
        if (len(main_path_points) > 0 and ball_position is not None) or use_fifth:
            # (Ramping for tilt_angle removed; tilt_angle is set directly below)
            # --- Update ball_position as the center of the yellow line BEFORE any tilt angle calculations ---
            hsv_img = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([35, 255, 255])
            mask_yellow = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
            lines = cv2.HoughLinesP(mask_yellow, 1, np.pi/180, threshold=20, minLineLength=20, maxLineGap=15)
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
            # Now proceed with tilt angle and navigation logic
            if use_fifth:
                target_xy = track_points[4]
            else:
                target_xy = main_path_points[current_target_idx]
            if ball_position is None:
                continue  # Skip this frame if no yellow line detected
            target_pos = np.array(target_xy)
            ball_pos = np.array(ball_position)
            # Distance to target (normalized and pixel)
            distance = np.linalg.norm(target_pos - ball_pos) / WARP_W
            pixel_distance = np.linalg.norm(target_pos - ball_pos)
            # Catchment area radius (in pixels)
            catchment_radius = 75
            slow_radius = 240
            # Arc calculation (as before)
            ball_dir = blue_dir
            v = ball_dir / (np.linalg.norm(ball_dir) + 1e-6)
            w = target_pos - ball_pos
            proj = np.dot(w, v) * v
            perp = w - proj
            # Tilt angle calculation
            direction_vec = target_pos - ball_pos
            blue_vec = blue_dir / (np.linalg.norm(blue_dir) + 1e-6)
            angle_to_target = math.atan2(direction_vec[1], direction_vec[0])
            blue_angle = math.atan2(blue_vec[1], blue_vec[0])
            tilt_angle = np.degrees(angle_to_target - blue_angle)
            # Normalize to [-180, 180]
            tilt_angle = (tilt_angle + 180) % 360 - 180
            tilt_angle = (tilt_angle) * 23 / 180  # Scale to max tilt of 23 degrees
            # Clamp to [-23, 23]
            tilt_angle = np.clip(tilt_angle, -23, 23)
            print(f"[DEBUG] direction_vec: {direction_vec}, blue_vec: {blue_vec}, tilt_angle: {tilt_angle:.2f}")
            # --- Reversing mode detection: moving away from the arc center (not just the target) ---
            if prev_ball_pos is not None:
                velocity_vec = ball_pos - prev_ball_pos
                # Use arc center if available, else fallback to target
                arc_ref_center = center if 'center' in locals() else target_pos
                prev_dist_to_arc = np.linalg.norm(arc_ref_center - prev_ball_pos)
                curr_dist_to_arc = np.linalg.norm(arc_ref_center - ball_pos)
                # Only activate reversing mode if ball is moving away from the arc and has moved >50 pixels further from the arc center
                #if (curr_dist_to_arc - prev_dist_to_arc > 10):
                    #reversing_mode = True
                #else:
                    #reversing_mode = False
            prev_ball_pos = ball_pos.copy()
            # Draw arc (normal or reversed)
            if not reversing_mode:
                if np.linalg.norm(perp) < 1e-6:
                    cv2.line(warped, tuple(ball_position), tuple(target_xy), (0,255,255), 3)
                    arc_tangent = v
                else:
                    r = np.dot(w, w) / (2 * np.linalg.norm(perp))
                    normal = np.array([-v[1], v[0]])
                    if np.dot(perp, normal) < 0:
                        normal = -normal
                    center = ball_pos + normal * r
                    radius = np.linalg.norm(center - ball_pos)
                    draw_arc_segment(warped, center, radius, ball_position, target_xy, (0,0,255), 3)
                    arc_tangent = np.array([-(ball_pos-center)[1], (ball_pos-center)[0]])
                    arc_tangent = arc_tangent / (np.linalg.norm(arc_tangent) + 1e-6)
            else:
                # Reversing mode: draw arc from the opposite direction
                w_rev = -w
                if np.linalg.norm(perp) < 1e-6:
                    cv2.line(warped, tuple(ball_position), tuple(target_xy), (0,255,0), 3)
                    arc_tangent = -v
                else:
                    r = np.dot(w_rev, w_rev) / (2 * np.linalg.norm(perp))
                    normal = np.array([-v[1], v[0]])
                    if np.dot(perp, normal) < 0:
                        normal = -normal
                    center = ball_pos + normal * r
                    radius = np.linalg.norm(center - ball_pos)
                    draw_arc_segment(warped, center, radius, ball_position, target_xy, (0,255,0), 3)
                    arc_tangent = -np.array([-(ball_pos-center)[1], (ball_pos-center)[0]])
                    arc_tangent = arc_tangent / (np.linalg.norm(arc_tangent) + 1e-6)
            # --- Control logic using arc tangent ---
            if pixel_distance <= catchment_radius:
                if use_fifth:
                    print("Ball reached 5th point (triggered by color==3). Returning to main path.")
                    current_target_idx = 0
                else:
                    print(f"Ball reached track point {current_target_idx+1} (within catchment radius). Cycling to next point.\n")
                    # Cycle to next point (wrap around to 0 after 3)
                    current_target_idx = (current_target_idx + 1) % len(main_path_points)
                    print(f"Now targeting track point {current_target_idx+1}.")
            # Speed ramps up if far, slows down if close
            speed = 2 * (2 / (1 + np.exp(-8 * (distance - 0.2))) - 1)
            # Always move in the direction of the arc tangent
            if np.dot(target_pos - ball_pos, arc_tangent) < 0:
                speed = -abs(speed)
            else:
                speed = abs(speed)
            if pixel_distance <= slow_radius:
                speed = 1 * (pixel_distance - catchment_radius) / (slow_radius - catchment_radius)
                speed = np.clip(speed, 0, 1) * np.sign(speed)
            speed = np.clip(speed, -1, 1)
            # Tilt angle calculation
            # --- User-requested tilt angle calculation ---
            direction_vec = target_pos - ball_pos
            blue_vec = blue_dir / (np.linalg.norm(blue_dir) + 1e-6)
            angle_to_target = math.atan2(direction_vec[1], direction_vec[0])
            blue_angle = math.atan2(blue_vec[1], blue_vec[0])
           # tilt_angle = np.degrees(angle_to_target - blue_angle)
            # Normalize to [-180, 180]
           # tilt_angle = (tilt_angle + 180) % 360 - 180
            #tilt_angle = (tilt_angle) * 23 / 180  # Scale to max tilt of 23 degrees
            # Clamp to [-23, 23]
           #+ tilt_angle = np.clip(tilt_angle, -23, 23)
            # print(f"[DEBUG] direction_vec: {direction_vec}, blue_vec: {blue_vec}, tilt_angle: {tilt_angle:.2f}")
            # --- Previous method commented out for reference ---
            # arc_tangent_norm = arc_tangent / (np.linalg.norm(arc_tangent) + 1e-6)
            # v_norm = v / (np.linalg.norm(v) + 1e-6)
            # dot = np.dot(v_norm, arc_tangent_norm)
            # det = v_norm[0]*arc_tangent_norm[1] - v_norm[1]*arc_tangent_norm[0]
            # angle_rad = math.atan2(det, dot)
            # angle_deg = np.degrees(angle_rad)
            # tilt_angle = np.clip((angle_deg / 90.0) * 23, -23, 23)
            # Flip speed and tilt if in reversing mode
            #if reversing_mode:
                #speed = -speed
                #tilt_angle = tilt_angle
            # --- Blue Line Detection and Arc Calculation ---
        hsv_img = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        mask_yellow = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
        lines = cv2.HoughLinesP(mask_yellow, 1, np.pi/180, threshold=20, minLineLength=20, maxLineGap=15)
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
                if len(track_points) > 0:
                    target_xy = track_points[current_target_idx]
                    # --- Ball targeting logic using track points and arc ---
                    if ball_position is not None:
                        target_pos = np.array(target_xy)
                        ball_pos = np.array(ball_position)
                        # Distance to target (normalized and pixel)
                        distance = np.linalg.norm(target_pos - ball_pos) / WARP_W
                        pixel_distance = np.linalg.norm(target_pos - ball_pos)
                        # Catchment area radius (in pixels)
                        catchment_radius = 75
                        slow_radius = 240
                        # Arc calculation (as before)
                        ball_dir = blue_dir
                        v = ball_dir / (np.linalg.norm(ball_dir) + 1e-6)
                        w = target_pos - ball_pos
                        proj = np.dot(w, v) * v
                        perp = w - proj
                        if np.linalg.norm(perp) < 1e-6:
                            # If the target is directly in line, just draw a straight line
                            cv2.line(warped, tuple(ball_position), tuple(target_xy), (0,255,255), 3)
                            arc_tangent = v
                            arc_center = None
                            arc_radius = None
                        else:
                            # Find the center of the circle passing through ball_pos and target_pos, with tangent v at ball_pos
                            r = np.dot(w, w) / (2 * np.linalg.norm(perp))
                            normal = np.array([-v[1], v[0]])
                            if np.dot(perp, normal) < 0:
                                normal = -normal
                            center = ball_pos + normal * r
                            radius = np.linalg.norm(center - ball_pos)
                            # Draw the arc from ball_position to target_pos
                            draw_arc_segment(warped, center, radius, ball_position, target_xy, (0,0,255), 3)
                            # Tangent at current position is perpendicular to (center - ball_pos)
                            arc_tangent = np.array([-(ball_pos-center)[1], (ball_pos-center)[0]])
                            arc_tangent = arc_tangent / (np.linalg.norm(arc_tangent) + 1e-6)
                            arc_center = center
                            arc_radius = radius
                        # --- Control logic using arc tangent ---
                        # Speed ramps up if far, slows down if close
                        speed = 2 * (2 / (1 + np.exp(-8 * (distance - 0.2))) - 1)
                        # Always move in the direction of the arc tangent
                        if np.dot(target_pos - ball_pos, arc_tangent) < 0:
                            speed = -abs(speed)
                        else:
                            speed = abs(speed)
                        if pixel_distance <= slow_radius:
                            speed = 1 * (pixel_distance - catchment_radius) / (slow_radius - catchment_radius)
                            speed = np.clip(speed, 0, 1) * np.sign(speed)
                        speed = np.clip(speed, -1, 1)
                        speed = -speed
                        tilt_angle = -tilt_angle
                        # Tilt angle calculation
                        # (Old tilt_angle calculation commented out; see main arc logic for actual value)
                        # arc_tangent_norm = arc_tangent / (np.linalg.norm(arc_tangent) + 1e-6)
                        # v_norm = v / (np.linalg.norm(v) + 1e-6)
                        # angle_to_target = math.atan2(arc_tangent_norm[1], arc_tangent_norm[0])
                        # blue_angle = math.atan2(v_norm[1], v_norm[0])
                        # tilt_angle_raw = np.degrees(angle_to_target - blue_angle)
                        # if tilt_angle_raw > 180:
                        #     tilt_angle_raw -= 360
                        # elif tilt_angle_raw < -180:
                        #     tilt_angle_raw += 360
                        # print(f"[DEBUG] arc_tangent: {arc_tangent_norm}, v: {v_norm}, raw: {tilt_angle_raw:.2f}, scaled: {tilt_angle:.2f}")
                        # Always send UDP for all points
                        # --- Control logic using arc tangent ---
                        
                        if pixel_distance <= catchment_radius:
                            print(f"Ball reached track point {current_target_idx+1}. Stopping.\n")
                            speed = 0
                            # Move to next point if available
                            if current_target_idx < len(track_points) - 2:
                                current_target_idx += 1
                                print(f"Now targeting track point {current_target_idx+1}.")
                            else:
                                current_target_idx = 0
                                print("All track points reached. Restarting from point 1.")
                    # Now always use the clicked points as the navigation target for UDP logic
                    # Throttle UDP sending to 10 times per second (every 0.1 seconds)
                    if 'last_udp_send_time' not in globals():
                        last_udp_send_time = 0
                    now = time.time()
                    #sign = np.sign(tilt_angle)
                   # tilt_angle = 23-abs(tilt_angle)
                    #tilt_angle = tilt_angle * sign
                    if now - last_udp_send_time >= 0.1:
                        udp_array = np.array([speed, tilt_angle, color_num], dtype=np.float64)
                        sock_send.sendto(udp_array.tobytes(), (UDP_IP_SEND, UDP_PORT))
                        print(f"[UDP] Sent speed: {speed:.2f}, tilt_angle: {tilt_angle:.2f}, color: {color_num}")
                        last_udp_send_time = now
        else:
            cv2.polylines(warped, [np.int32(dst_rect)], isClosed=True, color=(0,255,0), thickness=2)
        cv2.imshow("Live Tracking", warped)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()