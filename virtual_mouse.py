import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Get screen size
screen_w, screen_h = pyautogui.size()

# Smoothen cursor movement
prev_x, prev_y = 0, 0
smooth_factor = 7

while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark
            h, w, _ = frame.shape
            index_finger = landmarks[8]
            thumb_finger = landmarks[4]

            # Convert to screen coordinates
            x = int(index_finger.x * w)
            y = int(index_finger.y * h)
            screen_x = np.interp(x, [0, w], [0, screen_w])
            screen_y = np.interp(y, [0, h], [0, screen_h])

            # Smoothen the movement
            curr_x = prev_x + (screen_x - prev_x) / smooth_factor
            curr_y = prev_y + (screen_y - prev_y) / smooth_factor
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            # Detect click gesture
            thumb_x = int(thumb_finger.x * w)
            thumb_y = int(thumb_finger.y * h)
            distance = np.hypot(thumb_x - x, thumb_y - y)

            if distance < 30:
                pyautogui.click()
                time.sleep(0.2)

    # Show the webcam feed
    cv2.imshow("Virtual Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
