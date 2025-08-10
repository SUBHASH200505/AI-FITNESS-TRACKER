import cv2
import mediapipe as mp
import numpy as np
import time
from gtts import gTTS
from playsound import playsound
import os

class SquatTracker:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

        self.counter = 0
        self.stage = None
        self.session_active = False
        self.session_start_time = 0

        self.last_pose_time = time.time()
        self.pose_detected = True
        self.warned_user = False  # To avoid repeating audio warning

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180.0 else angle

    def speak(self, text):
        try:
            tts = gTTS(text=text, lang='en')
            filename = "temp_audio.mp3"
            tts.save(filename)
            playsound(filename)
            os.remove(filename)
        except:
            print("Audio Error")

    def count_fingers(self, hand_landmarks):
        tips_ids = [4, 8, 12, 16, 20]
        fingers = []

        if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
            fingers.append(1)
        else:
            fingers.append(0)

        for tip_id in tips_ids[1:]:
            if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)

        return sum(fingers)

    def detect_gesture(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                fingers_up = self.count_fingers(hand_landmarks)
                self.mp_drawing.draw_landmarks(image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                if fingers_up == 2:
                    return "start"
                elif fingers_up == 5:
                    return "stop"
        return None

    def track_squat(self, image, landmarks):
        hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
               landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                 landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        angle = self.calculate_angle(hip, knee, ankle)

        if angle > 160:
            self.stage = "up"
        if angle < 90 and self.stage == "up":
            self.stage = "down"
            self.counter += 1
            self.speak(str(self.counter))

        return image

    def update_display(self, image):
        feedback = "Work Hard!" if self.counter < 10 else \
                   "Keep Going!" if self.counter < 20 else \
                   "Doing Great!" if self.counter < 30 else "Excellent!"

        elapsed = int(time.time() - self.session_start_time) if self.session_active else 0

        cv2.putText(image, f'Squats: {self.counter}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.putText(image, feedback, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, f'Time: {elapsed}s', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        if self.session_active and not self.pose_detected and time.time() - self.last_pose_time > 5:
            cv2.putText(image, "Adjust your position for better tracking", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return image

    def end_session_feedback(self):
        if self.counter < 10:
            msg = "You can do better! Try to push more next time."
        elif self.counter < 20:
            msg = "Good effort! You're getting stronger!"
        elif self.counter < 30:
            msg = "Great job! Keep it up!"
        else:
            msg = "Excellent work! You're a squat master!"

        self.speak(f"Session ended. You completed {self.counter} squats. {msg}")

    def process_frame(self, frame):
        gesture = self.detect_gesture(frame)

        if gesture == "start" and not self.session_active:
            self.session_active = True
            self.counter = 0
            self.session_start_time = time.time()
            self.speak("Session started")

        elif gesture == "stop" and self.session_active:
            self.session_active = False
            self.end_session_feedback()

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if results.pose_landmarks:
            self.pose_detected = True
            self.last_pose_time = time.time()
            self.warned_user = False  # Reset warning
            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            if self.session_active:
                frame = self.track_squat(frame, results.pose_landmarks.landmark)
        else:
            if self.session_active:
                self.pose_detected = False
                if not self.warned_user and time.time() - self.last_pose_time > 5:
                    self.speak("Please come back in front of the camera for better tracking.")
                    self.warned_user = True

        return self.update_display(frame)

def main():
    cap = cv2.VideoCapture(0)
    tracker = SquatTracker()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = tracker.process_frame(frame)

        cv2.imshow('Squat Tracker (2 fingers = Start, 5 fingers = Stop, Q = Quit)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            if tracker.session_active:
                tracker.end_session_feedback()
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
