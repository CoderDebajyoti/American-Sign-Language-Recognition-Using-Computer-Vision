import cv2
import mediapipe as mp
import math
import pickle
import os

count = 0
label = input('Enter the label: ')
# Initialize the video capture
cap = cv2.VideoCapture(0)

if not os.path.isfile('dataset.pkl'):
    with open('dataset.pkl', 'wb') as file:
        pickle.dump({'data':[],'labels':[]},file)

with open('dataset.pkl', 'rb') as file:
    dataset = pickle.load(file)
    print(type(dataset))
with mp.solutions.hands.Hands(static_image_mode=False,max_num_hands=100,min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

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
            for hand_landmarks in results.multi_hand_landmarks:
                bone_angles = []
                for connection in mp.solutions.hands.HAND_CONNECTIONS:
                    bone_angles.append(math.atan2((hand_landmarks.landmark[connection[0]].x-hand_landmarks.landmark[connection[1]].x), (hand_landmarks.landmark[connection[0]].y-hand_landmarks.landmark[connection[1]].y)))
                dataset['data'].append(bone_angles)
                dataset['labels'].append(label)
                print(bone_angles)

        # Display the output
        cv2.imshow('Hand Detection', image)

        # Exit the loop if the 'q' key is pressed
        count += 1
        if (cv2.waitKey(1) & 0xFF == ord('q')) or count == 1000:
            with open('dataset.pkl', 'wb') as file:
                pickle.dump(dataset, file)
            break

cap.release()
cv2.destroyAllWindows()