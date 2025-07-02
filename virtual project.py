import cv2
import mediapipe as mp
import pyautogui
import math
import time

# Initialize camera and hand detector
cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands(max_num_hands=1)
drawing_utils = mp.solutions.drawing_utils

# Screen size
screen_width, screen_height = pyautogui.size()

# Parameters
click_threshold = 40
last_click_time = 0

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
            landmarks = hand.landmark

            # Get landmark positions
            index_tip = landmarks[8]
            thumb_tip = landmarks[4]
            middle_tip = landmarks[12]
            ring_tip = landmarks[16]

            # Convert normalized coords to pixels
            index_x = int(index_tip.x * frame_width)
            index_y = int(index_tip.y * frame_height)
            thumb_x = int(thumb_tip.x * frame_width)
            thumb_y = int(thumb_tip.y * frame_height)
            middle_x = int(middle_tip.x * frame_width)
            middle_y = int(middle_tip.y * frame_height)
            ring_y = int(ring_tip.y * frame_height)

            # Draw fingertip markers
            cv2.circle(frame, (index_x, index_y), 10, (0, 255, 255), -1)
            cv2.circle(frame, (thumb_x, thumb_y), 10, (0, 255, 0), -1)
            cv2.circle(frame, (middle_x, middle_y), 10, (255, 0, 0), -1)

            # Move mouse with index finger
            screen_x = screen_width / frame_width * index_x
            screen_y = screen_height / frame_height * index_y
            pyautogui.moveTo(screen_x, screen_y)

            current_time = time.time()

            # Left click: index + thumb
            distance_left = math.hypot(index_x - thumb_x, index_y - thumb_y)
            if distance_left < click_threshold and current_time - last_click_time > 1:
                pyautogui.click()
                last_click_time = current_time

            # Right click: middle + thumb
            distance_right = math.hypot(middle_x - thumb_x, middle_y - thumb_y)
            if distance_right < click_threshold and current_time - last_click_time > 1:
                pyautogui.click(button='right')
                last_click_time = current_time

            # Scroll gesture: ring finger vs middle finger height
            scroll_threshold = 20
            if abs(ring_y - middle_y) > scroll_threshold:
                if ring_y < middle_y:
                    pyautogui.scroll(20)   # Scroll up
                    print("Scrolling up")
                else:
                    pyautogui.scroll(-20)  # Scroll down
                    print("Scrolling down")

    cv2.imshow("Virtual Mouse", frame)
    cv2.waitKey(1)
