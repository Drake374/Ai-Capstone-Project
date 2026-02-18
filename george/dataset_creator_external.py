import cv2
import os
import numpy as np
import mss

name = "marcello_hernandez"
save_path = f"C:/Users/georg/OneDrive/Desktop/centennial/2026 winter sem/comp385/facenet_mtcnn/dataset/train/{name}"
os.makedirs(save_path, exist_ok=True)

# Initialize MSS (screen capture)
with mss.mss() as sct:

    # List monitors:
    # monitor[0] = virtual full desktop
    # monitor[1] = primary monitor
    # monitor[2+] = external monitors
    print("Available monitors:", sct.monitors)

    monitor_number = 2   # <-- CHANGE THIS to your external monitor
    monitor = sct.monitors[monitor_number]

    for i in range(20):
        # Capture screen
        screenshot = sct.grab(monitor)

        # Convert to numpy array
        frame = np.array(screenshot)

        # MSS gives BGRA → convert to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Save image
        cv2.imwrite(f"{save_path}/{i}.jpg", frame)

        # Show capture
        cv2.imshow("capture", frame)
        cv2.waitKey(200)

cv2.destroyAllWindows()
