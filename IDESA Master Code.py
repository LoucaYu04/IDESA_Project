import cv2
import cv2.aruco as aruco
import numpy as np
import time
import socket
import math

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
marker_size = 83
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

# Define a processing rate
processing_period = 0.25

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

cap = cv2.VideoCapture(1)
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
corner_ids = [2, 0, 5, 6]  # Replace with your actual corner marker IDs
inner_ids = [1, 8]           # Replace with your actual inner marker IDs

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
            # Prepare data for Simulink
            inner_positions = {}
            for iid in inner_ids:
                if iid in marker_centers:
                    x, y = marker_centers[iid]
                    inner_positions[iid] = (x / WARP_W, y / WARP_H)
                    cv2.circle(warped, tuple(np.int32(marker_centers[iid])), 8, (0,0,255), -1)
                    cv2.putText(warped, f"({x/WARP_W:.2f}, {y/WARP_H:.2f})", tuple(np.int32(marker_centers[iid]+10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            # --- Red Ball Detection and Tracking ---
            ball_position = None
            if 'ball_path_warped' not in globals():
                ball_path_warped = []
            # Only track if both ArUco markers are found
            if len(inner_positions) == 2:
                # Find red ball in the warped image
                hsv_ball = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
                lower_red1 = np.array([0, 120, 70])
                upper_red1 = np.array([10, 255, 255])
                lower_red2 = np.array([170, 120, 70])
                upper_red2 = np.array([180, 255, 255])
                mask_red = cv2.inRange(hsv_ball, lower_red1, upper_red1) | cv2.inRange(hsv_ball, lower_red2, upper_red2)
                mask_red = cv2.medianBlur(mask_red, 7)
                contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    # Filter all contours by area and circularity
                    min_area = 200
                    max_area = 5000
                    best_circularity = 0
                    best_contour = None
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if area < min_area or area > max_area:
                            continue
                        perimeter = cv2.arcLength(cnt, True)
                        if perimeter == 0:
                            continue
                        circularity = 4 * math.pi * (area / (perimeter * perimeter))
                        if circularity > 0.7 and circularity > best_circularity:
                            best_circularity = circularity
                            best_contour = cnt
                    if best_contour is not None:
                        (x_ball, y_ball), radius = cv2.minEnclosingCircle(best_contour)
                        center_ball = (int(x_ball), int(y_ball))
                        ball_position = center_ball
                        # --- Ball path tracking ---
                        ball_path_warped.append(center_ball)
                        if len(ball_path_warped) > 1000:
                            ball_path_warped = ball_path_warped[-1000:]
                        # Draw the path
                        if len(ball_path_warped) > 1:
                            cv2.polylines(warped, [np.array(ball_path_warped, dtype=np.int32)], False, (0,0,255), 2)
                        # Draw direction arrow
                        if len(ball_path_warped) > 5:
                            pt1 = ball_path_warped[-6]
                            pt2 = ball_path_warped[-1]
                            cv2.arrowedLine(warped, pt1, pt2, (0,255,255), 3, tipLength=0.3)
                        # Draw ball and lines to ArUco markers
                        cv2.circle(warped, center_ball, int(radius), (0,0,255), 2)
                        for iid in inner_ids:
                            if iid in marker_centers:
                                cv2.line(warped, center_ball, tuple(np.int32(marker_centers[iid])), (0,0,255), 2)
                        # Map ball position to normalized coordinates
                        x_norm, y_norm = x_ball / WARP_W, y_ball / WARP_H
                        cv2.putText(warped, f"Ball ({x_norm:.2f}, {y_norm:.2f})", (center_ball[0]+10, center_ball[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

                        # --- Arc and radius calculation ---
                        # Get ArUco marker positions
                        p1 = np.array(marker_centers[inner_ids[0]])
                        p2 = np.array(marker_centers[inner_ids[1]])
                        p_ball = np.array([x_ball, y_ball])
                        # Function to find circle center and radius from 3 points
                        def calc_circle(p1, p2, p3):
                            temp = p2 - p1
                            temp2 = p3 - p1
                            a = np.linalg.norm(p2 - p3)
                            b = np.linalg.norm(p1 - p3)
                            c = np.linalg.norm(p1 - p2)
                            # Calculate circle center
                            A = np.array([
                                [2*(p2[0]-p1[0]), 2*(p2[1]-p1[1])],
                                [2*(p3[0]-p1[0]), 2*(p3[1]-p1[1])]
                            ])
                            B = np.array([
                                p2[0]**2 + p2[1]**2 - p1[0]**2 - p1[1]**2,
                                p3[0]**2 + p3[1]**2 - p1[0]**2 - p1[1]**2
                            ])
                            try:
                                center = np.linalg.solve(A, B)
                            except np.linalg.LinAlgError:
                                center = np.array([np.nan, np.nan])
                            radius = np.linalg.norm(center - p1)
                            return center, radius

                        circle_center, circle_radius = calc_circle(p1, p2, p_ball)
                        # Calculate angles for arc length
                        def angle_between(center, pt):
                            return math.atan2(pt[1]-center[1], pt[0]-center[0])
                        angle1 = angle_between(circle_center, p_ball)
                        angle2 = angle_between(circle_center, p2)
                        arc_angle = abs(angle2 - angle1)
                        # Ensure arc_angle is in [0, pi]
                        arc_angle = min(arc_angle, 2*math.pi - arc_angle)
                        arc_length = abs(circle_radius * arc_angle)

                        # Draw circle and arc
                        if not np.isnan(circle_center[0]):
                            cv2.circle(warped, tuple(np.int32(circle_center)), int(circle_radius), (255,0,0), 1)
                        # Send ball position, arc length, and radius to Simulink
                        msg = f"ball,{x_norm:.4f},{y_norm:.4f};arc_length,{arc_length:.4f};arc_radius,{circle_radius:.4f};" + \
                            ";".join([f"{iid},{inner_positions[iid][0]:.4f},{inner_positions[iid][1]:.4f}" for iid in inner_ids])
                        # sock_send.sendto(msg.encode(), (UDP_IP_SEND, UDP_PORT))
                    else:
                        # If no ball, just send ArUco positions
                        msg = ";".join([f"{iid},{inner_positions[iid][0]:.4f},{inner_positions[iid][1]:.4f}" for iid in inner_ids])
                        # sock_send.sendto(msg.encode(), (UDP_IP_SEND, UDP_PORT))
                else:
                    # If no ball, just send ArUco positions
                    msg = ";".join([f"{iid},{inner_positions[iid][0]:.4f},{inner_positions[iid][1]:.4f}" for iid in inner_ids])
                    # sock_send.sendto(msg.encode(), (UDP_IP_SEND, UDP_PORT))
        else:
            cv2.polylines(warped, [np.int32(dst_rect)], isClosed=True, color=(0,255,0), thickness=2)
        cv2.imshow("Live Tracking", warped)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()