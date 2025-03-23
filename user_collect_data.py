import cv2
import mediapipe as mp
import csv
import math
import os
import pickle
import sys

# Get label from command-line args
if len(sys.argv) < 2:
    print("❌ No label provided.")
    sys.exit()

sign_label = sys.argv[1]
print(f"🖐️ Collecting data for: {sign_label}")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

max_data = 100
data_entries = []
labels = []

# Setup video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam.")
    sys.exit()

# Setup data dir
DATA_DIR = './user_data'
os.makedirs(DATA_DIR, exist_ok=True)
csv_filename = os.path.join(DATA_DIR, 'user_asl_data.csv')

# Open CSV file for writing
write_headers = not os.path.exists(csv_filename)
with open(csv_filename, 'a', newline='') as f:
    writer = csv.writer(f)
    if write_headers:
        writer.writerow(["label"] + [f"x{i},y{i}" for i in range(21)])

    data_per_label = 0

    while data_per_label < max_data:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

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

                writer.writerow([sign_label] + landmarks)
                data_entries.append(landmarks)
                labels.append(sign_label)
                data_per_label += 1

        cv2.putText(frame, f"Sign: {sign_label} ({data_per_label}/{max_data})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("ASL Data Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("❌ Collection interrupted early.")
            break

cap.release()
cv2.destroyAllWindows()

# Save collected data

# Merge with existing data if file exists
if os.path.exists('user_data.pickle'):
    with open('user_data.pickle', 'rb') as f:
        existing_data = pickle.load(f)
    data_entries = existing_data['user_data'] + data_entries
    labels = existing_data['labels'] + labels

# Save updated data
with open('user_data.pickle', 'wb') as f:
    pickle.dump({'user_data': data_entries, 'labels': labels}, f)

print(f"✅ Total samples saved: {len(data_entries)} for {len(set(labels))} unique labels.")




#with open('user_data.pickle', 'wb') as f:
    #pickle.dump({'user_data': data_entries, 'labels': labels}, f)
