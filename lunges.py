import cv2
import mediapipe as mp
import numpy as np
import time
import random
from gtts import gTTS
from playsound import playsound
import threading
import os
import uuid
import queue

# Global cooldown and queue for speak
speak_queue = queue.Queue()
last_spoken_time = 0

def speak(text):
    speak_queue.put(text)

def speak_worker():
    while True:
        text = speak_queue.get()
        if text is None:
            break
        try:
            filename = f"{uuid.uuid4().hex}.mp3"
            tts = gTTS(text=text, lang='en')
            tts.save(filename)
            playsound(filename)
            os.remove(filename)
        except Exception as e:
            print("Speak error:", e)
        speak_queue.task_done()

threading.Thread(target=speak_worker, daemon=True).start()

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(np.degrees(radians))
    return angle if angle <= 180 else 360 - angle

# Mediapipe setup
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

counter = 0
stage = None
session_started = False
last_gesture_time = 0
last_pose_feedback_time = 0
feedbacks = ['Great job!', 'Try harder!', 'Good form!', 'Keep it up!', 'Stay focused!']
motivations = ['Push yourself!', 'Never give up!', 'Be strong!', 'Consistency is key!']

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
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

        height, width, _ = img.shape

        # HAND GESTURE - Show 1 finger to start/end session
        if results_hands.multi_hand_landmarks:
            hand_landmarks = results_hands.multi_hand_landmarks[0]
            finger_up = []

            tips = [8, 12, 16, 20]  # Index to pinky
            for tip in tips:
                if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
                    finger_up.append(1)
                else:
                    finger_up.append(0)

            if finger_up == [1, 0, 0, 0]:  # Only index finger up
                if time.time() - last_gesture_time > 3:
                    if not session_started:
                        speak("Session started")
                        session_started = True
                        counter = 0
                        stage = None
                    else:
                        speak("Session ended")
                        speak(f"You completed {counter} repetitions")
                        time.sleep(0.5)
                        speak(random.choice(feedbacks))
                        time.sleep(0.5)
                        speak(random.choice(motivations))
                        session_started = False
                        counter = 0
                        stage = None
                    last_gesture_time = time.time()

        # No person detected
        if not results_pose.pose_landmarks:
            cv2.putText(img, 'Adjust your position', (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            if session_started and time.time() - last_pose_feedback_time > 5:
                speak("Adjust your position")
                last_pose_feedback_time = time.time()
            cv2.imshow("Lunges Counter", img)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        # Draw pose landmarks
        mp_drawing.draw_landmarks(img, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = results_pose.pose_landmarks.landmark

        # Get left leg coordinates
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        angle = calculate_angle(left_hip, left_knee, left_ankle)

        # Rep counter logic
        if session_started:
            if angle < 85 and stage != "down":
                stage = "down"
            elif angle > 165 and stage == "down":
                stage = "up"
                counter += 1
                speak(f"Rep {counter} completed")

            # Display reps
            cv2.putText(img, f'Reps: {counter}', (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(img, 'Show 1 to End', (30, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        else:
            cv2.putText(img, 'Show 1 to Start', (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)

        cv2.imshow("Lunges Counter", img)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

cap.release()
cv2.destroyAllWindows()
speak_queue.put(None)  # Stop the speak worker thread
