import sys
import time
from pathlib import Path

import cv2

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.camera_stream import VideoStream

def main():
    url = "rtsp://admin:Haithy123@192.168.1.87:554/onvif1"
    print(f"Connecting to {url}...")
    
    # Initialize background camera stream (resize to 1280x720 to save CPU)
    cam = VideoStream(url, resize_dim=(1280, 720)).start()
    time.sleep(1.0) # Wait for camera to warm up

    print("View camera stream. Press ESC to exit.")
    
    while True:
        ret, frame = cam.read()
        
        if not ret or frame is None:
            continue

        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) == 27: # ESC
            break

    cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()