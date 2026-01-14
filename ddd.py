import pygame
import socket
import numpy as np

# --- UDP Setup ---
UDP_IP_SEND = "138.38.228.211"
UDP_PORT = 25000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# --- Pygame Setup ---
pygame.init()
screen = pygame.display.set_mode((200, 200))  # Needed for keyboard input
pygame.joystick.init()

# --- Joystick Detection ---
if pygame.joystick.get_count() == 0:
    print("No joystick detected!")
    exit()

joystick = pygame.joystick.Joystick(0)
joystick.init()
print("Xbox controller connected:", joystick.get_name())

print("\nReading triggers + JoyToKey steering...\n")

# --- Main Loop ---
while True:
    pygame.event.pump()
    keys = pygame.key.get_pressed()

    # -------------------------
    #   TRIGGER INPUT (RT/LT)
    # -------------------------
    rt = joystick.get_axis(5)
    lt = joystick.get_axis(4)

    # Normalize triggers to [0, 1]
    if rt < -0.1 or lt < -0.1:
        rt_val = (rt + 1) / 2
        lt_val = (lt + 1) / 2
    else:
        rt_val = rt
        lt_val = lt

    move_val = rt_val - lt_val   # Forward/backward throttle

    # -------------------------
    #   STEERING INPUT (JoyToKey)
    # -------------------------
    steer_val = 0

    # Left (G, H, J → -6, -12, -20)
    if keys[pygame.K_h]:
        steer_val = -5
    elif keys[pygame.K_j]:
        steer_val = -10

    # Right (V, B, N → 6, 12, 20)
    if keys[pygame.K_b]:
        steer_val = 5
    elif keys[pygame.K_n]:
        steer_val = 10

    # -------------------------
    #   PRINT + UDP SEND
    # -------------------------
    print(f"move={move_val:.3f}, steer={steer_val}")

    arr = np.array([move_val, steer_val], dtype=np.float64)
    sock.sendto(arr.tobytes(), (UDP_IP_SEND, UDP_PORT))

    pygame.time.wait(50)
