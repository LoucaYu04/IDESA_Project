import cv2
import numpy as np
import socket
import time
import tkinter as tk
from tkinter import messagebox

# --- UDP Setup ---
UDP_IP_SEND = "138.38.228.211"
UDP_IP_RECEIVE = "172.26.109.96" #LOUCA'S LAPTOP IP
UDP_PORT = 25000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

'''
# --- Camera Setup ---
cap = cv2.VideoCapture(1)  # Use 0 or the correct index for Arducam
# Disable auto white balance and set manual white balance if supported
cap.set(cv2.CAP_PROP_AUTO_WB, 0)
cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4000)  # You can adjust this value for your lighting

# --- Color Ranges (HSV) ---
# Adjust these ranges as needed for your robot's colors
red_lower1 = np.array([0, 100, 100])
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([160, 100, 100])
red_upper2 = np.array([180, 255, 255])
green_lower = np.array([40, 70, 70])
green_upper = np.array([80, 255, 255])
blue_lower = np.array([100, 150, 70])
blue_upper = np.array([130, 255, 255])

last_color_id = 0
last_color_time = 0
sent_after_delay = False
DELAY_SECONDS = 2.5  # 2-3 seconds

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not found!")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red mask (handle wrap-around in HSV)
    mask_red = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)
    mask_green = cv2.inRange(hsv, green_lower, green_upper)
    mask_blue = cv2.inRange(hsv, blue_lower, blue_upper)

    # Count pixels for each color
    red_count = np.sum(mask_red > 0)
    green_count = np.sum(mask_green > 0)
    blue_count = np.sum(mask_blue > 0)

    # Determine dominant color
    color_id = 0
    if max(red_count, green_count, blue_count) == 0:
        color_name = "None"
    elif red_count >= green_count and red_count >= blue_count:
        color_id = 1
        color_name = "Red"
    elif green_count >= red_count and green_count >= blue_count:
        color_id = 2
        color_name = "Green"
    else:
        color_id = 3
        color_name = "Blue"

    now = time.time()
    # Track last detected color and time
    if color_id > 0:
        if color_id != last_color_id:
            last_color_id = color_id
            last_color_time = now
            sent_after_delay = False
    else:
        last_color_id = 0
        last_color_time = 0
        sent_after_delay = False

    # --- Assign mask for dominant color ---
    if color_id == 1:
        mask = mask_red
    elif color_id == 2:
        mask = mask_green
    elif color_id == 3:
        mask = mask_blue
    else:
        mask = None

    # --- Find and draw contour around largest color area ---
    if mask is not None and np.any(mask > 0):
        # Find largest contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 100:  # Only draw if area is significant
                x, y, w_box, h_box = cv2.boundingRect(largest)
                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 255), 2)
                cv2.putText(frame, 'Color Area', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Show result
    cv2.putText(frame, f"Detected: {color_name}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    cv2.imshow("Robot Color Detection", frame)

    # Send UDP message 2-3 seconds after color is seen
    if last_color_id > 0 and not sent_after_delay and now - last_color_time >= DELAY_SECONDS:
        if last_color_id == 2:  # Green
            msg = bytes([1])
            print("[UDP] Sending: 1 (Green)")
            sock.sendto(msg, (UDP_IP_SEND, UDP_PORT))
            sent_after_delay = True
        elif last_color_id == 3:  # Blue
            msg = bytes([2])
            print("[UDP] Sending: 2 (Blue)")
            sock.sendto(msg, (UDP_IP_SEND, UDP_PORT))
            sent_after_delay = True
        elif last_color_id == 1:  # Red
            msg = bytes([3])
            print("[UDP] Sending: 3 (Red)")
            sock.sendto(msg, (UDP_IP_SEND, UDP_PORT))
            sent_after_delay = True

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''

root = tk.Tk()
root.title("Mars Robot Race Control")
root.geometry("400x320")
root.configure(bg="#1a1a2e")

label = tk.Label(root, text="Send Mars Race Control Flag", font=("Arial", 16, "bold"), fg="#f5f6fa", bg="#1a1a2e")
label.pack(pady=25)

btn_green = tk.Button(root, text="Green Flag", width=20, height=2, bg="#21e675", fg="#222", font=("Arial", 12, "bold"), activebackground="#16a34a", activeforeground="#fff", command=lambda: send_udp_for_color("Green"))
btn_green.pack(pady=10)

btn_blue = tk.Button(root, text="Blue Flag", width=20, height=2, bg="#3a7bd5", fg="#fff", font=("Arial", 12, "bold"), activebackground="#27408b", activeforeground="#fff", command=lambda: send_udp_for_color("Blue"))
btn_blue.pack(pady=10)

btn_red = tk.Button(root, text="Red Flag", width=20, height=2, bg="#e94560", fg="#fff", font=("Arial", 12, "bold"), activebackground="#b22234", activeforeground="#fff", command=lambda: send_udp_for_color("Red"))
btn_red.pack(pady=10)

# Add a color box to show the last sent color
color_box = tk.Label(root, text="", width=20, height=2, bg="#444", relief="ridge", bd=3)
color_box.pack(pady=10)

# --- UI for color selection ---
def send_udp_for_color(color_name):
    if color_name == "Green":
        msg = bytes([1])
        print("[UDP] Sending: 1 (Green)")
        color_box.config(bg="#21e675")
    elif color_name == "Blue":
        msg = bytes([2])
        print("[UDP] Sending: 2 (Blue)")
        color_box.config(bg="#3a7bd5")
    elif color_name == "Red":
        msg = bytes([3])
        print("[UDP] Sending: 3 (Red)")
        color_box.config(bg="#e94560")
    else:
        return
    sock.sendto(msg, (UDP_IP_SEND, UDP_PORT))

footer = tk.Label(root, text="Mission Control: Mars Signal Uplink", font=("Arial", 10, "italic"), fg="#aaa", bg="#1a1a2e")
footer.pack(side="bottom", pady=10)

root.mainloop()
