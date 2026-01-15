import pygame
import socket
import numpy as np
import tkinter as tk
from threading import Thread
import os
os.environ["SDL_JOYSTICK_RAWINPUT"] = "1"
os.environ["SDL_JOYSTICK_HIDAPI"] = "0"


# --- UDP Setup ---
UDP_IP_SEND = "138.38.227.25"
UDP_IP_RECEIVE = "172.26.109.96" #LOUCA'S LAPTOP IP
UDP_PORT = 25000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# --- Pygame Joystick Setup ---
pygame.init()
pygame.joystick.init()
if pygame.joystick.get_count() == 0:
    print("No joystick detected!")
    exit()
joystick = pygame.joystick.Joystick(0)
joystick.init()

print("Xbox controller connected.")

# --- Tkinter UI Setup ---
root = tk.Tk()
root.title("Xbox Controller UDP Monitor")
root.geometry("400x220")
root.configure(bg="#222")

status_label = tk.Label(root, text="Controller: Not Connected", font=("Arial", 14, "bold"), fg="#fff", bg="#222")
status_label.pack(pady=10)

values_label = tk.Label(root, text="Values sent: --, --", font=("Arial", 12), fg="#fff", bg="#222")
values_label.pack(pady=10)

# --- Controller Thread ---
def controller_loop():
    global joystick
    connected = False
    while True:
        pygame.event.pump()
        if not connected and pygame.joystick.get_count() > 0:
            joystick = pygame.joystick.Joystick(0)
            joystick.init()
            status_label.config(text="Controller: Connected", fg="#21e675")
            connected = True
        elif pygame.joystick.get_count() == 0:
            status_label.config(text="Controller: Not Connected", fg="#e94560")
            connected = False
            values_label.config(text="Values sent: --, --")
            pygame.time.wait(200)
            continue

        # Triggers: RT (axis 5), LT (axis 2) on most Xbox controllers
        rt = joystick.get_axis(5)
        lt = joystick.get_axis(4)
        # Remap triggers from [0, 1] (not pressed to fully pressed)
        # Some controllers use [0, 1], others use [-1, 1]. Try to detect and remap if needed.
        if rt < -0.1 or lt < -0.1:  # If triggers are in [-1, 1] range
            rt_val = (rt + 1) / 2
            lt_val = (lt + 1) / 2
        else:  # If triggers are already in [0, 1] range
            rt_val = rt
            lt_val = lt
        # First value: RT - LT, range -1 to 1, 0 when both not pressed
        move_val = rt_val - lt_val
        # Second value: Left joystick X axis (axis 0), -1 (left) to 1 (right), map to -20 to 20
        joy_x = joystick.get_axis(0)  # Right joystick X axis
        print(joy_x)
        steer_val = float(joy_x) *60.0  # -20 (left) to 20 (right)
        if abs(steer_val) < 6.6:
            steer_val = 0.0  # Deadzone:
        arr = np.array([move_val, steer_val], dtype=np.float64)
        sock.sendto(arr.tobytes(), (UDP_IP_SEND, UDP_PORT))
        values_label.config(text=f"Values sent: {move_val:.3f}, {steer_val:.3f}")
        pygame.time.wait(100)

# Run controller loop in a thread so UI stays responsive
Thread(target=controller_loop, daemon=True).start()

root.mainloop()
