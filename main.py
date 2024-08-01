import cv2
import mediapipe as mp
import math
import joblib

model = joblib.load('trained_model.joblib')
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize the video capture
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a later selfie-view display
        image = cv2.flip(image, 1)

        # Convert the image from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Perform hand detection
        results = hands.process(image)

        # Draw the hand annotations on the image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                bone_angles = []
                if handedness.classification[0].label != 'Right':
                    for landmark in hand_landmarks.landmark:
                        landmark.x *= -1
                for connection in mp.solutions.hands.HAND_CONNECTIONS:
                    bone_angles.append(math.atan2((hand_landmarks.landmark[connection[0]].x-hand_landmarks.landmark[connection[1]].x), (hand_landmarks.landmark[connection[0]].y-hand_landmarks.landmark[connection[1]].y)))
                prediction = model.predict([bone_angles])
                cv2.putText(image, prediction[0], (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the output
        cv2.imshow('Hand Detection', image)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()