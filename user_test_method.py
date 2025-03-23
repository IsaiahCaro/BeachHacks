import cv2
import mediapipe as mp
import pickle
import numpy as np

# Load the trained model
model_dict = pickle.load(open('user_model.p', 'rb'))
model = model_dict['user_model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.3)

# Check if labels are numeric or string
labels_dict = {'A': 'A', 'B': 'B', 'C': 'C'}  # If your model predicts letters directly

cap = cv2.VideoCapture(0)

while True:
    data_aux = []
    ret, frame = cap.read()
    
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Normalize landmarks
            wrist = hand_landmarks.landmark[0]
            base_distance = np.linalg.norm(
                [hand_landmarks.landmark[0].x - hand_landmarks.landmark[9].x,
                 hand_landmarks.landmark[0].y - hand_landmarks.landmark[9].y]
            )

            for lm in hand_landmarks.landmark:
                norm_x = (lm.x - wrist.x) / base_distance
                norm_y = (lm.y - wrist.y) / base_distance
                data_aux.extend([norm_x, norm_y])

        # Ensure correct input size
        if len(data_aux) == 42:  # 21 landmarks * 2 (x, y)
            prediction = model.predict([np.asarray(data_aux)])

            # Check if model outputs strings or integers
            if isinstance(prediction[0], str):  # Model predicts letters directly
                predicted_character = prediction[0]
            else:  # Model predicts numbers, map to letters
                predicted_character = labels_dict.get(int(prediction[0]), "?")

            print(f"Predicted Letter: {predicted_character}")

            # Display prediction on the frame
            cv2.putText(frame, predicted_character, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('ASL Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
