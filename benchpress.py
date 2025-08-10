import cv2
import mediapipe as mp
import numpy as np
import time
from gtts import gTTS
import os
from playsound import playsound

class BenchPressTracker:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.9, min_tracking_confidence=0.9)

        self.counter = 0
        self.stage = None
        self.session_started = False
        self.lost_tracking = False
        self.start_time = None
        self.end_time = None
        self.last_gesture_time = 0
        self.last_gesture = None

    def speak(self, text):
        try:
            tts = gTTS(text=text, lang='en')
            filename = "voice.mp3"
            tts.save(filename)
            playsound(filename)
            os.remove(filename)
        except Exception as e:
            print(f"Error speaking: {e}")

    def count_fingers(self, frame):
        """Detect number of fingers shown (used for gesture control only)"""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        count = 0
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            landmarks = hand.landmark
            tips = [4, 8, 12, 16, 20]
            if landmarks[4].x < landmarks[3].x:  # Thumb
                count += 1
            for tip in tips[1:]:
                if landmarks[tip].y < landmarks[tip - 2].y:
                    count += 1
            self.mp_drawing.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)
        return count

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def process_pose(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        if not results.pose_landmarks:
            if self.session_started and not self.lost_tracking:
                self.lost_tracking = True
                self.speak("Please adjust to be visible on screen.")
            return frame
        else:
            self.lost_tracking = False

        self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        landmarks = results.pose_landmarks.landmark

        # Get key joints
        l_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                      landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        l_elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                   landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        l_wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                   landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        r_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        r_elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                   landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        r_wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                   landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        # Calculate angles
        left_angle = self.calculate_angle(l_shoulder, l_elbow, l_wrist)
        right_angle = self.calculate_angle(r_shoulder, r_elbow, r_wrist)
        avg_angle = (left_angle + right_angle) / 2

        # Count reps
        if avg_angle > 160:
            self.stage = "up"
        if avg_angle < 90 and self.stage == "up":
            self.stage = "down"
            self.counter += 1
            self.speak(f"Rep {self.counter} counted!")

        # Draw info
        cv2.putText(frame, f'Angle: {int(avg_angle)}', (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f'Reps: {self.counter}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(frame, f"Stage: {self.stage}", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        return frame

    def handle_gestures(self, frame):
        current_time = time.time()
        if current_time - self.last_gesture_time > 1:
            fingers = self.count_fingers(frame)
            self.last_gesture_time = current_time

            # Start session
            if not self.session_started and fingers == 5:
                self.session_started = True
                self.counter = 0
                self.stage = None
                self.start_time = time.time()
                self.speak("Workout started. Let's go!")

            # Stop session
            elif self.session_started and fingers == 2:
                self.session_started = False
                self.end_time = time.time()
                total_time = int(self.end_time - self.start_time)
                self.speak(f"Workout complete. You did {self.counter} reps in {total_time} seconds.")
                if self.counter < 10:
                    self.speak("Nice try! Keep pushing!")
                elif 10 <= self.counter < 20:
                    self.speak("Great job! You're getting stronger!")
                else:
                    self.speak("Awesome work! You crushed it!")
                self.counter = 0

    def run(self):
        cap = cv2.VideoCapture(0)
        self.speak("Show five fingers to start workout. Show two fingers to stop.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            # Handle gestures regardless of session
            self.handle_gestures(frame)

            # If session active, process pose and count reps
            if self.session_started:
                frame = self.process_pose(frame)

            cv2.imshow('Bench Press Tracker', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                if self.session_started:
                    self.end_time = time.time()
                    total_time = int(self.end_time - self.start_time)
                    self.speak(f"Workout ended. You completed {self.counter} reps in {total_time} seconds.")
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = BenchPressTracker()
    tracker.run()
