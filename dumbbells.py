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

# Cooldown to prevent fast speaking
last_gesture_time = 0
last_position_feedback_time = 0  # Time tracker for position feedback

def speak(text):
    # Speech function to speak the given text
    def _speak():
        try:
            filename = f"{uuid.uuid4().hex}.mp3"
            tts = gTTS(text=text, lang='en')
            tts.save(filename)
            playsound(filename)
            os.remove(filename)
        except Exception as e:
            print("Speak error:", e)

    threading.Thread(target=_speak).start()  # Use threading to prevent blocking other actions

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    return angle if angle <= 180 else 360 - angle

# Setup MediaPipe
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

counter = 0
stage = None
session_started = False
session_ended = False

# Motivation messages
motivation_messages = [
    "Great job! Keep going!",
    "You're doing awesome! Stay strong!",
    "Keep it up! Almost there!",
    "Well done! You're crushing it!"
]

def give_feedback_and_motivation():
    # Provide text feedback
    feedback_text = f"Session ended. You completed {counter} reps. "
    motivation_text = random.choice(motivation_messages)
    
    # Speak the feedback and motivation sequentially
    speak(feedback_text)  # Feedback about session end
    time.sleep(3)  # Wait for 3 seconds before giving motivation (to avoid overlap)
    speak(motivation_text)  # Motivation after feedback

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

        # Gesture detection (1 finger to start/end)
        finger_up = [0, 0, 0, 0]
        if results_hands.multi_hand_landmarks:
            hand_landmarks = results_hands.multi_hand_landmarks[0]
            tips = [8, 12, 16, 20]
            for i, tip in enumerate(tips):
                if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
                    finger_up[i] = 1
                else:
                    finger_up[i] = 0

        # Detect the "2" gesture (two fingers up) to start/end the session
        if finger_up == [1, 1, 0, 0] and time.time() - last_gesture_time > 3:
            if not session_started:
                speak("Session started")
                session_started = True
                session_ended = False
                counter = 0
                stage = None
            elif session_started and not session_ended:
                give_feedback_and_motivation()  # Provide feedback and motivation at the end of session
                session_ended = True
                session_started = False
                counter = 0
                stage = None
            last_gesture_time = time.time()

        # No pose detected
        if not results_pose.pose_landmarks:
            if time.time() - last_position_feedback_time > 5:  # Only speak every 5 seconds if no pose detected
                speak('Adjust your position')
                last_position_feedback_time = time.time()
            cv2.putText(img, 'Adjust your position', (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.imshow("Dumbbell Curl Counter", img)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        # Draw landmarks
        mp_drawing.draw_landmarks(img, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = results_pose.pose_landmarks.landmark

        # Right arm joints
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

        # Debug: print the angle for feedback
        print(f"Angle: {angle}")

        # Curl rep logic
        if session_started:
            if angle > 160:
                stage = "down"
            if angle < 45 and stage == "down":
                stage = "up"
                counter += 1

                # Provide voice feedback for each rep
                speak(f"Rep {counter}")

            # Display rep count
            cv2.putText(img, f'Reps: {counter}', (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(img, 'Show 2 to End', (30, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        else:
            cv2.putText(img, 'Show 2 to Start', (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)

        cv2.imshow("Dumbbell Curl Counter", img)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

cap.release()
cv2.destroyAllWindows()
