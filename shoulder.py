import cv2
import mediapipe as mp
import numpy as np
import time
import random
from gtts import gTTS
import pygame
import threading
import os
import uuid

pygame.mixer.init()
last_gesture_time = 0

def speak(text):
    def _speak():
        try:
            filename = f"{uuid.uuid4().hex}.mp3"
            tts = gTTS(text=text, lang='en')
            tts.save(filename)
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            os.remove(filename)
        except Exception as e:
            print("Speak error:", e)
    threading.Thread(target=_speak).start()

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(np.degrees(radians))
    return angle if angle <= 180 else 360 - angle

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

counter = 0
stage = None
session_started = False
session_ended = False

motivation_messages = [
    "Excellent job! Keep pushing!",
    "You're strong! Keep going!",
    "Amazing effort!",
    "You're doing great, don't stop!"
]

def give_feedback_and_motivation():
    feedback_text = f"Session ended. You completed {counter} shoulder presses."
    motivation_text = random.choice(motivation_messages)
    speak(feedback_text)
    time.sleep(2)
    speak(motivation_text)

with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose, \
     mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        results_pose = pose.process(img)
        results_hands = hands.process(img)
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        finger_up = [0, 0, 0, 0]
        if results_hands.multi_hand_landmarks:
            hand_landmarks = results_hands.multi_hand_landmarks[0]
            tips = [8, 12, 16, 20]
            for i, tip in enumerate(tips):
                if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
                    finger_up[i] = 1
                else:
                    finger_up[i] = 0

        if finger_up == [1, 1, 0, 0] and time.time() - last_gesture_time > 3:
            if not session_started:
                speak("Shoulder press session started")
                session_started = True
                session_ended = False
                counter = 0
                stage = None
            elif session_started and not session_ended:
                give_feedback_and_motivation()
                session_started = False
                session_ended = True
                counter = 0
                stage = None
            last_gesture_time = time.time()

        if not results_pose.pose_landmarks:
            cv2.putText(img, 'Make sure full body is visible', (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Shoulder Press Counter", img)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        mp_drawing.draw_landmarks(img, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = results_pose.pose_landmarks.landmark

        # Get required points
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        # Calculate angles for both arms
        right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

        avg_angle = (right_angle + left_angle) / 2

        # Show angle values
        cv2.putText(img, f'R: {int(right_angle)} L: {int(left_angle)}', (30, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if session_started:
            if avg_angle > 160:
                stage = "down"
            if avg_angle < 70 and stage == "down":
                stage = "up"
                counter += 1
                speak(f"Rep {counter}")

            cv2.putText(img, f'Reps: {counter}', (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(img, 'Show 2 fingers to End', (30, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        else:
            cv2.putText(img, 'Show 2 fingers to Start', (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)

        cv2.imshow("Shoulder Press Counter", img)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
