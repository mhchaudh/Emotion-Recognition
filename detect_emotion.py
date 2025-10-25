import os
import json
import cv2
import numpy as np
import time
from collections import deque
try:
    import tensorflow as tf
except Exception as e:
    raise ImportError("TensorFlow import failed. Install tensorflow (e.g. pip install tensorflow).") from e

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'emotion_model.h5')
LABELS_PATH = os.path.join(BASE_DIR, 'labels.json')


if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
    raise FileNotFoundError("Model or labels.json not found. Run train_model.py first and ensure files are in the project folder.")


model = tf.keras.models.load_model(MODEL_PATH)
with open(LABELS_PATH, 'r') as f:
    labels = json.load(f)  

input_shape = model.input_shape  
if input_shape and len(input_shape) >= 3:
    TARGET_H = int(input_shape[1])
    TARGET_W = int(input_shape[2])
else:
    TARGET_H, TARGET_W = 48, 48  


cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")


trackers = {}     
next_id = 0
PROB_QUEUE = 8
MAX_AGE = 1.5
DIST_THRESH = 80

def center_of_box(box):
    x, y, w, h = box
    return (int(x + w/2), int(y + h/2))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    current_time = time.time()
    used_ids = set()

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        try:
            roi = cv2.resize(roi_gray, (TARGET_W, TARGET_H))
        except Exception:
            continue
        roi = roi.astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)  
        preds = model.predict(roi, verbose=0)[0]  

  
        c = center_of_box((x, y, w, h))
        matched_id = None
        min_dist = float('inf')
        for tid, info in trackers.items():
            tc = info['center']
            dist = np.hypot(tc[0]-c[0], tc[1]-c[1])
            if dist < min_dist and dist < DIST_THRESH:
                min_dist = dist
                matched_id = tid

        if matched_id is None:
           
            matched_id = next_id
            next_id += 1
            trackers[matched_id] = {
                'center': c,
                'probs': deque(maxlen=PROB_QUEUE),
                'last_seen': current_time
            }


        trackers[matched_id]['center'] = c
        trackers[matched_id]['probs'].append(preds)
        trackers[matched_id]['last_seen'] = current_time
        used_ids.add(matched_id)

        
        probs_stack = np.stack(trackers[matched_id]['probs'], axis=0)
        mean_probs = probs_stack.mean(axis=0)
        idx = int(np.argmax(mean_probs))
        label = labels.get(str(idx), labels.get(idx, 'Unknown'))

        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{label}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    
    stale = [tid for tid,info in trackers.items() if current_time - info['last_seen'] > MAX_AGE]
    for tid in stale:
        del trackers[tid]

    cv2.imshow('Emotion Recognition - press q to quit', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
