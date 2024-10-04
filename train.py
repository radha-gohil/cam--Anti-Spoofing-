import math
import time

import cv2
import cvzone
from ultralytics import YOLO

confidence = 0.6
blink_threshold = 5  # Adjust this threshold as needed for blinking detection

cap = cv2.VideoCapture(0)  # For Webcam
cap.set(3, 640)
cap.set(4, 480)
# cap = cv2.VideoCapture("../Videos/motorbikes.mp4")  # For Video

model = YOLO("../models/best.pt")

classNames = ["fake", "real"]

prev_frame_time = 0
new_frame_time = 0

blink_counter = 0
last_blink_time = time.time()

no_movement_time_threshold = 5  # Threshold for no movement in seconds
last_movement_time = time.time()

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate time since last movement
    elapsed_time = new_frame_time - last_movement_time

    # Detect faces using YOLO
    results = model(img, stream=True, verbose=False)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            if conf > confidence:
                if classNames[cls] == 'real':
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)

                cvzone.cornerRect(img, (x1, y1, w, h), colorC=color, colorR=color)
                cvzone.putTextRect(img, f'{classNames[cls].upper()} {int(conf * 100)}%',
                                   (max(0, x1), max(35, y1)), scale=2, thickness=4, colorR=color,
                                   colorB=color)

                # Update last movement time
                last_movement_time = new_frame_time

    # Check for no movement condition
    if elapsed_time > no_movement_time_threshold:
        # Implement action for no movement detection (e.g., display a message or alert)
        cv2.putText(img, "No movement detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
