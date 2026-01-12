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
corner_ids = [2, 3, 5, 6]  # Replace with your actual corner marker IDs
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
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break
        frame = gray_world_correction(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        marker_centers = {}
        if ids is not None:
            ids_flat = ids.flatten()
            aruco.drawDetectedMarkers(frame, corners, ids)
            for i, marker_id in enumerate(ids_flat):
                c = corners[i][0]
                center = np.mean(c, axis=0)
                marker_centers[marker_id] = center
            # Draw the fixed area
            cv2.polylines(frame, [np.int32(corner_centers)], isClosed=True, color=(0,255,0), thickness=2)
            dst_pts = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=np.float32)
            H, _ = cv2.findHomography(corner_centers, dst_pts)
            # Prepare data for Simulink
            inner_positions = {}
            for iid in inner_ids:
                if iid in marker_centers:
                    pt = np.array([*marker_centers[iid], 1.0])
                    mapped = H @ pt
                    mapped /= mapped[2]
                    x, y = mapped[0], mapped[1]
                    inner_positions[iid] = (x, y)
                    cv2.circle(frame, tuple(np.int32(marker_centers[iid])), 8, (0,0,255), -1)
                    cv2.putText(frame, f"({x:.2f}, {y:.2f})", tuple(np.int32(marker_centers[iid]+10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            # --- Red Ball Detection and Tracking ---
            ball_position = None
            # Persistent path for the red ball
            if 'ball_path' not in globals():
                ball_path = []
            # Only track if both ArUco markers are found
            if len(inner_positions) == 2:
                # Find red ball in the frame
                hsv_ball = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                lower_red1 = np.array([0, 120, 70])
                upper_red1 = np.array([10, 255, 255])
                lower_red2 = np.array([170, 120, 70])
                upper_red2 = np.array([180, 255, 255])
                mask_red = cv2.inRange(hsv_ball, lower_red1, upper_red1) | cv2.inRange(hsv_ball, lower_red2, upper_red2)
                # Morphological operations to clean up mask
                mask_red = cv2.medianBlur(mask_red, 7)
                contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(largest_contour) > 100:
                        (x_ball, y_ball), radius = cv2.minEnclosingCircle(largest_contour)
                        center_ball = (int(x_ball), int(y_ball))
                        ball_position = center_ball
                        # --- Ball path tracking ---
                        ball_path.append(center_ball)
                        if len(ball_path) > 1000:
                            ball_path = ball_path[-1000:]
                        # Draw the path
                        if len(ball_path) > 1:
                            cv2.polylines(frame, [np.array(ball_path, dtype=np.int32)], False, (0,0,255), 2)
                        # Draw direction arrow
                        if len(ball_path) > 5:
                            pt1 = ball_path[-6]
                            pt2 = ball_path[-1]
                            cv2.arrowedLine(frame, pt1, pt2, (0,255,255), 3, tipLength=0.3)
                        # Draw ball and lines to ArUco markers
                        cv2.circle(frame, center_ball, int(radius), (0,0,255), 2)
                        for iid in inner_ids:
                            cv2.line(frame, center_ball, tuple(np.int32(marker_centers[iid])), (0,0,255), 2)
                        # Map ball position to normalized coordinates
                        pt_ball = np.array([x_ball, y_ball, 1.0])
                        mapped_ball = H @ pt_ball
                        mapped_ball /= mapped_ball[2]
                        x_norm, y_norm = mapped_ball[0], mapped_ball[1]
                        cv2.putText(frame, f"Ball ({x_norm:.2f}, {y_norm:.2f})", (center_ball[0]+10, center_ball[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                        # Send ball position to Simulink
                        msg = f"ball,{x_norm:.4f},{y_norm:.4f};" + ";".join([f"{iid},{inner_positions[iid][0]:.4f},{inner_positions[iid][1]:.4f}" for iid in inner_ids])
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
            cv2.polylines(frame, [np.int32(corner_centers)], isClosed=True, color=(0,255,0), thickness=2)
        cv2.imshow("Live Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()