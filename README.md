import cv2
import mediapipe as mp
import math



mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

webcam = cv2.VideoCapture(0)
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

def calculate_distance(point1, point2):
    """Calculate the 3D Euclidean distance between two points."""
    return math.sqrt(
        (point1.x - point2.x) ** 2 +
        (point1.y - point2.y) ** 2 +
        (point1.z - point2.z) ** 2
    )

def count_fingers(hand_landmarks):
    # Define landmarks for fingertips and wrist
    finger_tips = [4, 8, 12, 16, 20]
    wrist = hand_landmarks.landmark[0]

    # Set initial finger count to 5
    up_fingers = 5

    # Calculate total distance from wrist to each fingertip
    for tip_idx in finger_tips:
        tip = hand_landmarks.landmark[tip_idx]
        total_distance = calculate_distance(wrist, tip)

        # If the tip is within 75% of its max distance from wrist, count it as "down"
        if total_distance < 0.75 * calculate_distance(wrist, hand_landmarks.landmark[9]):  
            up_fingers -= 1

    return up_fingers

while webcam.isOpened():
    success, img = webcam.read()
    if not success:
        break

    #activate hand tracking 
    img = cv2.resize(img, (640, 480))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img)

    #drawings 
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, hand_landmarks, connections = mp_hands.HAND_CONNECTIONS)

            num_fingers_up = count_fingers(hand_landmarks)
            print(f"Fingers up: {num_fingers_up}")


    cv2.imshow('Koolac', img)
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

webcam.release()

cv2.destroyAllWindows()
