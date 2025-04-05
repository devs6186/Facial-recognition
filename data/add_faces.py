import cv2
import numpy as np
import os
import pickle

# Constants for pink box
TOP_LEFT = (118, 183)
BOTTOM_RIGHT = (1468, 1000)
BOX_WIDTH = BOTTOM_RIGHT[0] - TOP_LEFT[0]
BOX_HEIGHT = BOTTOM_RIGHT[1] - TOP_LEFT[1]

# Load background
background = cv2.imread("data/background.png")
if background is None:
    raise FileNotFoundError("Background image not found at data/background.png")
background = cv2.resize(background, (1920, 1080))

# Load or initialize dataset
faces = []
names = []
if os.path.exists('data/face_data.pkl'):
    with open('data/face_data.pkl', 'rb') as f:
        faces = pickle.load(f)
    with open('data/names.pkl', 'rb') as f:
        names = pickle.load(f)

# Load DNN face detector
face_net = cv2.dnn.readNetFromCaffe(
    "data/deploy.prototxt",
    "data/res10_300x300_ssd_iter_140000.caffemodel"
)

# Get user input
name = input("Enter your name: ").strip()
if not name:
    print("[ERROR] Name cannot be empty.")
    exit()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

count = 0
print(f"[INFO] Capturing face data for '{name}'... Look at the camera.")

while True:
    ret, full_frame = cap.read()
    if not ret:
        continue

    # Create display frame
    display_frame = background.copy()
    
    # Process webcam feed for pink box area
    cropped = full_frame[TOP_LEFT[1]:BOTTOM_RIGHT[1], TOP_LEFT[0]:BOTTOM_RIGHT[0]]
    frame = cv2.resize(cropped, (BOX_WIDTH, BOX_HEIGHT))

    # Face detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (x, y, x1, y1) = box.astype("int")
            
            face = frame[y:y1, x:x1]
            if face.size == 0:
                continue

            face_resized = cv2.resize(face, (50, 50)).astype("float32") / 255.0
            faces.append(face_resized.flatten())
            names.append(name)
            count += 1

            cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
            cv2.putText(frame, f'{name} ({count})', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Overlay processed frame
    display_frame[TOP_LEFT[1]:BOTTOM_RIGHT[1], TOP_LEFT[0]:BOTTOM_RIGHT[0]] = frame
    
    cv2.namedWindow("Adding Faces", cv2.WINDOW_NORMAL)
    cv2.imshow("Adding Faces", display_frame)
    
    if cv2.waitKey(1) == ord('q') or count >= 60:
        break

cap.release()
cv2.destroyAllWindows()

# Save data
with open('data/face_data.pkl', 'wb') as f:
    pickle.dump(faces, f)
with open('data/names.pkl', 'wb') as f:
    pickle.dump(names, f)

print(f"[INFO] Face data for '{name}' saved successfully. Total samples: {count}")