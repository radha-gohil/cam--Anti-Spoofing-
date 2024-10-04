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


def calculate_ear(eye_landmarks):
    if len(eye_landmarks) != 6:
        return -1  # EAR calculation not possible with less than 6 points

    # Calculate EAR using eye landmarks
    v1 = math.sqrt((eye_landmarks[1][0] - eye_landmarks[5][0]) ** 2 + (eye_landmarks[1][1] - eye_landmarks[5][1]) ** 2)
    v2 = math.sqrt((eye_landmarks[2][0] - eye_landmarks[4][0]) ** 2 + (eye_landmarks[2][1] - eye_landmarks[4][1]) ** 2)
    h = math.sqrt((eye_landmarks[0][0] - eye_landmarks[3][0]) ** 2 + (eye_landmarks[0][1] - eye_landmarks[3][1]) ** 2)

    return (v1 + v2) / (2 * h)


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

    # Detect eyes and calculate EAR for blinking detection
    eyes = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml').detectMultiScale(gray, scaleFactor=1.3,
                                                                                                 minNeighbors=5)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)

        # Get eye landmarks for EAR calculation
        eye_landmarks = [(ex + ew // 2, ey + eh // 6), (ex + ew // 2, ey + 5 * eh // 6),
                         (ex + ew // 6, ey + eh // 2), (ex + 5 * ew // 6, ey + eh // 2),
                         (ex + ew // 4, ey + eh // 4), (ex + 3 * ew // 4, ey + eh // 4)]

        ear = calculate_ear(eye_landmarks)

        if ear != -1 and ear < blink_threshold:
            blink_counter += 1
        else:
            blink_counter = 0

        if blink_counter >= 3:
            # Implement action for blinking detection (e.g., print a message)
            print("Blink detected!")

    # Check for no movement condition
    if elapsed_time > no_movement_time_threshold:
        # Implement action for no movement detection (e.g., display a message or alert)
        cv2.putText(img, "No movement detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Check if no blinking detected after displaying no movement message for 1 second
    if elapsed_time > no_movement_time_threshold + 1 and blink_counter < 3:
        # Implement action for declaring as fake (e.g., display a message or alert)
        cv2.putText(img, "Fake detected!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
