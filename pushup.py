import cv2
import mediapipe as mp
import numpy as np
import time
from enum import Enum
from gtts import gTTS
from playsound import playsound
import os

class ExerciseType(Enum):
    PUSHUP = "Push-up"

class ExerciseTracker:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8)
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )

        self.counter = 0
        self.stage = None
        self.session_active = False
        self.session_start_time = 0
        self.last_feedback_time = 0
        self.break_given = False
        self.screen_feedback_given = False

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle
        return angle

    def speak(self, text):
        tts = gTTS(text=text, lang='en')
        filename = "temp_voice.mp3"
        tts.save(filename)
        playsound(filename)
        os.remove(filename)

    def countdown(self, seconds=3):
        for i in range(seconds, 0, -1):
            self.speak(str(i))
            time.sleep(1)
        self.speak("Go!")

    def check_hand_gesture(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            fingers_up = []

            tips_ids = [4, 8, 12, 16, 20]

            for i, tip in enumerate(tips_ids):
                if i == 0:  # Thumb
                    fingers_up.append(1 if hand_landmarks.landmark[tip].x < hand_landmarks.landmark[tip - 1].x else 0)
                else:
                    fingers_up.append(1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y else 0)

            if fingers_up == [1, 1, 1, 1, 1]:  # All fingers
                return "start"
            elif fingers_up == [0, 1, 0, 0, 0]:  # Only index
                return "stop"
            elif fingers_up == [1, 1, 1, 0, 0]:  # Thumb, index, middle
                return "break"
        return None

    def track_pushup(self, image, landmarks):
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

        distance = abs(left_shoulder.x - right_shoulder.x)

        if distance < 0.15 and not self.screen_feedback_given:
            self.speak("Move closer to the camera")
            self.screen_feedback_given = True
        elif distance > 0.5 and not self.screen_feedback_given:
            self.speak("Move back from the camera")
            self.screen_feedback_given = True
        elif 0.15 <= distance <= 0.5:
            self.screen_feedback_given = False

        left = self.get_angle(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                              self.mp_pose.PoseLandmark.LEFT_ELBOW,
                              self.mp_pose.PoseLandmark.LEFT_WRIST)

        right = self.get_angle(landmarks, self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
                               self.mp_pose.PoseLandmark.RIGHT_ELBOW,
                               self.mp_pose.PoseLandmark.RIGHT_WRIST)

        avg_angle = (left + right) / 2

        if avg_angle > 160:
            self.stage = "up"
        if avg_angle < 90 and self.stage == "up":
            self.stage = "down"
            self.counter += 1
            self.speak(f"{self.counter}")
            if self.counter % 5 == 0:
                self.speak("Keep it up!")

        return image

    def get_angle(self, landmarks, a, b, c):
        p1 = [landmarks[a.value].x, landmarks[a.value].y]
        p2 = [landmarks[b.value].x, landmarks[b.value].y]
        p3 = [landmarks[c.value].x, landmarks[c.value].y]
        return self.calculate_angle(p1, p2, p3)

    def update_display(self, image):
        feedback = ""
        if self.counter < 10:
            feedback = "Work Hard!"
        elif self.counter < 20:
            feedback = "Keep Pushing!"
        elif self.counter < 30:
            feedback = "Doing Great!"
        else:
            feedback = "Excellent!"

        elapsed = int(time.time() - self.session_start_time) if self.session_active else 0

        cv2.putText(image, f'Reps: {self.counter}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(image, f'{feedback}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, f'Time: {elapsed}s', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        if elapsed > 60 and not self.break_given:
            self.speak("You've been working hard. Consider taking a break!")
            self.break_given = True

        return image

    def process_frame(self, frame, exercise_type):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(image_rgb)
        image = frame.copy()

        gesture = self.check_hand_gesture(image)

        if gesture == "start" and not self.session_active:
            self.session_active = True
            self.counter = 0
            self.session_start_time = time.time()
            self.break_given = False
            self.speak("Session will begin in")
            self.countdown()
            self.speak("Session started")

        elif gesture == "stop" and self.session_active:
            self.session_active = False
            self.speak(f"Session ended. You completed {self.counter} pushups. Great job!")

        elif gesture == "break" and self.session_active:
            self.speak("Taking a short break. Relax for 5 seconds.")
            time.sleep(5)
            self.speak("Break over. Resume your pushups!")

        if pose_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            if self.session_active and exercise_type == ExerciseType.PUSHUP:
                image = self.track_pushup(image, pose_results.pose_landmarks.landmark)

        image = self.update_display(image)
        return image

def main():
    cap = cv2.VideoCapture(0)
    tracker = ExerciseTracker()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = tracker.process_frame(frame, ExerciseType.PUSHUP)

        cv2.imshow('Smart Push-Up Tracker', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            if tracker.session_active:
                tracker.speak(f"Session ended. You completed {tracker.counter} pushups. Great job!")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
