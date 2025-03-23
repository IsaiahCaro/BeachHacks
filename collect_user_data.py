import sys
import cv2
import mediapipe as mp
import math
import os
from datetime import datetime

# Get label from args
if len(sys.argv) < 2:
    print("❌ Label not provided.")
    exit()
label = sys.argv[1]

# Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Failed to open webcam.")
    exit()

DATA_DIR = 'user_data'
os.makedirs(DATA_DIR, exist_ok=True)
csv_path = os.path.join(DATA_DIR, 'user_asl_data.csv')

count = 0
max_count = 100

print(f"📸 Starting data collection for '{label}'...")

while count < max_count:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            wrist = hand_landmarks.landmark[0]
            base_distance = math.hypot(
                wrist.x - hand_landmarks.landmark[9].x,
                wrist.y - hand_landmarks.landmark[9].y
            )
            landmarks = []
            for lm in hand_landmarks.landmark:
                norm_x = (lm.x - wrist.x) / base_distance
                norm_y = (lm.y - wrist.y) / base_distance
                landmarks.extend([norm_x, norm_y])
            with open(csv_path, 'a') as f:
                f.write(f"{label}," + ",".join(map(str, landmarks)) + "\n")
            count += 1

    cv2.putText(frame, f"{label} ({count}/{max_count})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Collecting ASL Data", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Interrupted early.")
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Done.")