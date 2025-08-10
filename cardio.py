import cv2
import mediapipe as mp
import numpy as np
import time
from gtts import gTTS
import os
import playsound

class CardioTracker:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

        self.start_time = None
        self.running_time = 0
        self.prev_hip_y = None
        self.moving = False
        self.session_started = False
        self.cardio_count = 0
        self.last_feedback_time = 0

    def detect_movement(self, landmarks):
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        avg_hip_y = (left_hip.y + right_hip.y) / 2

        if self.prev_hip_y is not None:
            movement = abs(avg_hip_y - self.prev_hip_y)
            if movement > 0.02:
                self.moving = True
                return True

        self.prev_hip_y = avg_hip_y
        self.moving = False
        return False

    def count_fingers(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        finger_count = 0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                tips = [4, 8, 12, 16, 20]  # Thumb to pinky tips
                for idx in tips[1:]:
                    if hand_landmarks.landmark[idx].y < hand_landmarks.landmark[idx - 2].y:
                        finger_count += 1
                # Thumb
                if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
                    finger_count += 1
                self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return finger_count

    def process_frame(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        fingers = self.count_fingers(frame)

        if fingers == 5 and not self.session_started:
            self.speak("Session started. Keep moving!")
            self.start_time = time.time()
            self.running_time = 0
            self.session_started = True
        elif fingers == 2 and self.session_started:
            self.end_session()

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

            if self.session_started and self.detect_movement(results.pose_landmarks.landmark):
                if self.start_time is not None:
                    current_time = time.time()
                    self.running_time += current_time - self.start_time
                    self.start_time = current_time

                # Voice feedback every 30 seconds
                if int(self.running_time) % 30 == 0 and int(self.running_time) != self.last_feedback_time:
                    minutes = int(self.running_time // 60)
                    seconds = int(self.running_time % 60)
                    self.speak(f"{minutes} minutes and {seconds} seconds. Keep going!")
                    self.last_feedback_time = int(self.running_time)

        else:
            if self.session_started:
                self.speak("Please adjust the screen for proper tracking.")
                self.start_time = None

        return self.update_display(frame)

    def end_session(self):
        self.session_started = False
        total_minutes = int(self.running_time // 60)
        total_seconds = int(self.running_time % 60)
        self.cardio_count += 1

        self.speak(f"Session ended. Great job!")
        self.speak(f"Total time: {total_minutes} minutes and {total_seconds} seconds.")
        self.speak(f"You have completed {self.cardio_count} cardio sessions.")

        feedback = self.get_feedback(self.running_time)
        self.speak(feedback)

        motivation = self.get_motivation()
        self.speak(motivation)

        self.running_time = 0

    def update_display(self, image):
        minutes = int(self.running_time // 60)
        seconds = int(self.running_time % 60)

        status = "Work Hard" if minutes < 3 else \
                 "Good Job" if 3 <= minutes < 10 else \
                 "You are Rocking"

        cv2.putText(image, f'Time: {minutes} min {seconds} sec', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, f'Status: {status}', (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f'Sessions: {self.cardio_count}', (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        return image

    def speak(self, text):
        print(f"Speaking: {text}")
        tts = gTTS(text=text, lang='en')
        tts.save("temp.mp3")
        playsound.playsound("temp.mp3", True)
        os.remove("temp.mp3")

    def get_feedback(self, time_elapsed):
        minutes = int(time_elapsed // 60)
        if minutes < 1:
            return "Try to keep going longer next time!"
        elif 1 <= minutes < 3:
            return "Nice start! Let’s aim for more next time!"
        elif 3 <= minutes < 10:
            return "Good consistency! You're getting stronger!"
        else:
            return "Excellent effort! You crushed it!"

    def get_motivation(self):
        messages = [
            "Every drop of sweat brings you closer to your goal.",
            "Your body can stand almost anything. It’s your mind you have to convince.",
            "You don’t have to be extreme, just consistent.",
            "Push yourself because no one else is going to do it for you.",
            "You are stronger than you think!"
        ]
        return np.random.choice(messages)

def main():
    cap = cv2.VideoCapture(0)
    tracker = CardioTracker()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = tracker.process_frame(frame)
        cv2.imshow('Cardio Running Tracker', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            if tracker.session_started:
                tracker.end_session()
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
