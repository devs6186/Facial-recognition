from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
from datetime import datetime
import csv
import os

# Constants for face box area (pink box)
TOP_LEFT = (118, 183)
BOTTOM_RIGHT = (1468, 1000)
BOX_WIDTH = BOTTOM_RIGHT[0] - TOP_LEFT[0]
BOX_HEIGHT = BOTTOM_RIGHT[1] - TOP_LEFT[1]

# Load background image
background = cv2.imread("data/background.png")
if background is None:
    raise FileNotFoundError("Background image not found at data/background.png")
background = cv2.resize(background, (1920, 1080))

# Load classifier and face data
with open('data/names.pkl', 'rb') as f:
    LABELS = pickle.load(f)
with open('data/face_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(FACES, LABELS)

# Load DNN face detector
face_net = cv2.dnn.readNetFromCaffe(
    "data/deploy.prototxt",
    "data/res10_300x300_ssd_iter_140000.caffemodel"
)

# Attendance setup
ATTENDANCE_FILE = 'data/attendance.csv'
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, 'w', newline='') as f:
        csv.writer(f).writerow(['Name', 'Date', 'Time'])

recognized_today = set()

def mark_attendance(name):
    today = datetime.now().strftime('%Y-%m-%d')
    time_now = datetime.now().strftime('%H:%M:%S')
    key = f"{name}_{today}"
    if key not in recognized_today:
        recognized_today.add(key)
        with open(ATTENDANCE_FILE, 'a', newline='') as f:
            csv.writer(f).writerow([name, today, time_now])
        return True
    return False

def draw_corners(frame, x, y, w, h, color=(120, 50, 200), thickness=2, length=30):
    cv2.line(frame, (x, y), (x + length, y), color, thickness)
    cv2.line(frame, (x, y), (x, y + length), color, thickness)
    cv2.line(frame, (x + w, y), (x + w - length, y), color, thickness)
    cv2.line(frame, (x + w, y), (x + w, y + length), color, thickness)
    cv2.line(frame, (x, y + h), (x, y + h - length), color, thickness)
    cv2.line(frame, (x, y + h), (x + length, y + h), color, thickness)
    cv2.line(frame, (x + w, y + h), (x + w - length, y + h), color, thickness)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - length), color, thickness)

def add_clock_overlay(img):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    cv2.putText(img, current_time, (img.shape[1] - 200, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# Start webcam
video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    ret, full_frame = video.read()
    if not ret:
        continue

    # Create overlay with background
    overlay = background.copy()
    
    # Extract and resize the pink box area
    cropped = full_frame[TOP_LEFT[1]:BOTTOM_RIGHT[1], TOP_LEFT[0]:BOTTOM_RIGHT[0]]
    frame = cv2.resize(cropped, (BOX_WIDTH, BOX_HEIGHT))
    add_clock_overlay(frame)

    # Face detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (x1, y1, x2, y2) = box.astype("int")
            w, h = x2 - x1, y2 - y1
            
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue
                
            face_input = cv2.resize(face, (50, 50)).astype("float32") / 255.0
            name = knn.predict([face_input.flatten()])[0]

            draw_corners(frame, x1, y1, w, h)
            cv2.rectangle(frame, (x1, y1 - 40), (x1 + w, y1), (120, 50, 200), -1)
            cv2.putText(frame, name, (x1 + 5, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if mark_attendance(name):
                cv2.putText(frame, "Attendance Marked", (x1 + 5, y2 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Overlay the processed frame
    overlay[TOP_LEFT[1]:BOTTOM_RIGHT[1], TOP_LEFT[0]:BOTTOM_RIGHT[0]] = frame
    
    # Display attendance count
    cv2.putText(overlay, f"Today's Attendance: {len(recognized_today)}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
    cv2.imshow("Face Recognition", overlay)
    
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()