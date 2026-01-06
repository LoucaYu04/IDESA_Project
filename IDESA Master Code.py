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

# --- Camera Preview Loop ---

# --- Functions to run for each color ---
def on_red_detected():
    print("Red detected! Running red function...")
    # Add your red event code here

def on_green_detected():
    print("Green detected! Running green function...")
    # Add your green event code here

def on_blue_detected():
    print("Blue detected! Running blue function...")
    # Add your blue event code here

prev_color = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from camera.")
        break


    # --- White balance correction (Gray World Assumption) ---
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

    # Detect ArUco markers in the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # Draw detected markers
    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)
        cv2.putText(frame, f"Detected {len(ids)} marker(s)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    else:
        cv2.putText(frame, "No ArUco markers detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # --- Color Detection ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Red mask (two ranges for HSV wraparound)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    # Green mask
    lower_green = np.array([36, 100, 100])
    upper_green = np.array([86, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Blue mask
    lower_blue = np.array([94, 80, 2])
    upper_blue = np.array([126, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)


    # Find which color is most present
    red_count = cv2.countNonZero(mask_red)
    green_count = cv2.countNonZero(mask_green)
    blue_count = cv2.countNonZero(mask_blue)

    color = None
    mask = None
    if red_count > green_count and red_count > blue_count and red_count > 500:
        color = 'red'
        mask = mask_red
    elif green_count > red_count and green_count > blue_count and green_count > 500:
        color = 'green'
        mask = mask_green
    elif blue_count > red_count and blue_count > green_count and blue_count > 500:
        color = 'blue'
        mask = mask_blue

    # Draw bounding box around largest area of detected color
    if mask is not None:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 500:
                x, y, w, h = cv2.boundingRect(largest_contour)
                color_map = {'red': (0,0,255), 'green': (0,255,0), 'blue': (255,0,0)}
                cv2.rectangle(frame, (x, y), (x+w, y+h), color_map[color], 3)

    # Run function if color changed
    if color and color != prev_color:
        if color == 'red':
            on_red_detected()
        elif color == 'green':
            on_green_detected()
        elif color == 'blue':
            on_blue_detected()
        prev_color = color

    # Show which color is detected
    if color:
        cv2.putText(frame, f"{color.capitalize()} detected", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    else:
        cv2.putText(frame, "No color detected", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow("Frame", frame)
    # Press 'q' to quit the preview
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()