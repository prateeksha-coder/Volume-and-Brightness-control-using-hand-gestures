

import mediapipe as mp
import cv2
import numpy as np
import subprocess
from math import hypot
import screen_brightness_control as sbc

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# ------------------ macOS Volume Control ------------------
def set_mac_volume(volume_percent):
    volume_percent = int(np.clip(volume_percent, 0, 100))
    subprocess.run(
        ["osascript", "-e", f"set volume output volume {volume_percent}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()


while True:
    success, img = cap.read()
    if not success:
        print("Failed to read frame.")
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[i].classification[0].label

            mp_draw.draw_landmarks(
                img, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            thumb_tip = hand_landmarks.landmark[
                mp_hands.HandLandmark.THUMB_TIP
            ]
            index_tip = hand_landmarks.landmark[
                mp_hands.HandLandmark.INDEX_FINGER_TIP
            ]

            h, w, _ = img.shape
            thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
            index_pos = (int(index_tip.x * w), int(index_tip.y * h))

            cv2.circle(img, thumb_pos, 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, index_pos, 10, (255, 0, 0), cv2.FILLED)
            cv2.line(img, thumb_pos, index_pos, (0, 255, 0), 3)

            distance = hypot(
                index_pos[0] - thumb_pos[0],
                index_pos[1] - thumb_pos[1]
            )

            
            if hand_label == "Right":
                volume_percent = np.interp(distance, [30, 300], [0, 100])
                set_mac_volume(volume_percent)

                vol_bar = np.interp(distance, [30, 300], [400, 150])
                cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 2)
                cv2.rectangle(img, (50, int(vol_bar)), (85, 400),
                              (255, 0, 0), cv2.FILLED)
                cv2.putText(
                    img,
                    f'Volume: {int(volume_percent)}%',
                    (40, 450),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    3
                )

            elif hand_label == "Left":
                brightness = np.interp(distance, [30, 300], [0, 100])
                try:
                    sbc.set_brightness(int(brightness))
                except Exception as e:
                    print("Brightness error:", e)

                bright_bar = np.interp(distance, [30, 300], [400, 150])
                cv2.rectangle(img, (100, 150), (135, 400), (0, 255, 0), 2)
                cv2.rectangle(img, (100, int(bright_bar)), (135, 400),
                              (0, 255, 0), cv2.FILLED)
                cv2.putText(
                    img,
                    f'Brightness: {int(brightness)}%',
                    (90, 450),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    3
                )

    cv2.imshow("Mac Gesture Volume & Brightness Controller", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
