import os
import pickle
import csv
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './user_data'
PICKLE_FILE = 'user_data.pickle'
CSV_FILE = 'user_asl_data.csv'

user_data = []
labels = []
csv_data = []
data_aux = []

# Process each image and extract hand landmarks
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        x_, y_ = [], []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)
                
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            user_data.append(data_aux)
            labels.append(dir_)
            csv_data.append([dir_] + data_aux)

# Save data to pickle
with open(PICKLE_FILE, 'wb') as f:
    pickle.dump({'user_data': user_data, 'labels': labels}, f)

# Save data to CSV
with open(CSV_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    header = ["Label"] + [f"X{i//2}" if i % 2 == 0 else f"Y{i//2}" for i in range(len(data_aux))]
    writer.writerow(header)
    writer.writerows(csv_data)

def visualize_hand_data(csv_filename):
    asl_signs = defaultdict(list)  # Dictionary to store hand landmarks by label

    with open(csv_filename, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Skip header
        
        for row in csv_reader:
            label = row[0]
            landmark_values = list(map(float, row[1:]))  # Convert values to float
            asl_signs[label].append(landmark_values)  # Store data under the corresponding label
    
    for label, hand_data_list in asl_signs.items():
        fig, ax = plt.subplots()
        
        for hand_data in hand_data_list:
            x_vals = hand_data[0::2]  # Extract x-coordinates
            y_vals = hand_data[1::2]  # Extract y-coordinates
            ax.scatter(x_vals, y_vals, label=label, marker='o')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f"ASL Sign: {label}")
        plt.legend()
        plt.show()

# Call visualization function
f = open('user_data.pickle', 'wb')
pickle.dump({'user_data': user_data, 'labels': labels})
