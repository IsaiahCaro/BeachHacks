import cv2
import mediapipe as mp
import csv
import math
import os
import pickle

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

# Maximum Data Per Character
max_data = 100

data_entries = []  # Array for landmark data
labels = []  # Array for corresponding labels

# Setup video capture
cap = cv2.VideoCapture(0)

# Get user input for the sign label
sign_label = input("Enter the ASL sign label: ")

data_per_label = 0  # Track data count for current label

# Create data directory if it doesn't exist
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

csv_filename = os.path.join(DATA_DIR, 'asl_data.csv')

# Open CSV file
with open(csv_filename, 'a', newline='') as f:
    writer = csv.writer(f)

    # Write headers only if the file is empty
    if os.stat(csv_filename).st_size == 0:
        writer.writerow(["label"] + [f"x{i},y{i}" for i in range(21)])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame for a mirrored effect
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract and normalize landmarks
                landmarks = []
                wrist = hand_landmarks.landmark[0]

                # Calculate hand size for scaling
                base_distance = math.sqrt((hand_landmarks.landmark[0].x - hand_landmarks.landmark[9].x)**2 +
                                          (hand_landmarks.landmark[0].y - hand_landmarks.landmark[9].y)**2)

                for lm in hand_landmarks.landmark:
                    # Normalize coordinates relative to the wrist
                    norm_x = (lm.x - wrist.x) / base_distance
                    norm_y = (lm.y - wrist.y) / base_distance
                    landmarks.extend([norm_x, norm_y])

                # Store normalized landmarks with the label
                writer.writerow([sign_label] + landmarks)
                data_entries.append(landmarks)
                labels.append(sign_label)  # Ensure label matches entry

                data_per_label += 1

                # Ask for a new label every 100 data points
                if data_per_label >= max_data:
                    sign_label = input("Enter the next ASL sign label: ")
                    data_per_label = 0  # Reset count for new label

        # Display the frame
        cv2.putText(frame, f"Sign: {sign_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('ASL Data Collection', frame)

        # Quit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Save data properly
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data_entries, 'labels': labels}, f)

# Load and check data
data_dict = pickle.load(open('data.pickle', 'rb'))
print(f"Collected {len(data_dict['data'])} samples with {len(set(data_dict['labels']))} unique labels.")
