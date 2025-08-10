import cv2
import mediapipe as mp
import time
import math
from gtts import gTTS
from playsound import playsound
import uuid
import os

class PullUpTracker:
    def __init__(self):
        # Pose Setup
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils

        # Hand Setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                                         min_detection_confidence=0.7, min_tracking_confidence=0.7)

        # Counters and flags
        self.counter = 0
        self.pull_up_complete = False
        self.last_spoken_count = -1

        self.started = False
        self.ended = False
        self.paused = False
        self.session_start_time = None
        self.last_seen_time = time.time()
        self.user_detected = True

        self.pullup_threshold = 50
        self.release_threshold = 160

    def speak(self, text):
        filename = f"voice_{uuid.uuid4()}.mp3"
        tts = gTTS(text=text, lang='en')
        tts.save(filename)
        playsound(filename)
        os.remove(filename)

    def calculate_angle(self, a, b, c):
        ab = [b[0] - a[0], b[1] - a[1]]
        bc = [c[0] - b[0], c[1] - b[1]]

        dot_product = ab[0] * bc[0] + ab[1] * bc[1]
        ab_len = math.hypot(ab[0], ab[1])
        bc_len = math.hypot(bc[0], bc[1])

        cosine_angle = dot_product / (ab_len * bc_len + 1e-6)
        angle = math.acos(max(min(cosine_angle, 1.0), -1.0))
        return math.degrees(angle)

    def update_counter(self):
        self.counter += 1
        if self.counter != self.last_spoken_count:
            self.speak(str(self.counter))
            self.last_spoken_count = self.counter

    def count_fingers(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(image_rgb)
        if not result.multi_hand_landmarks:
            return 0

        hand_landmarks = result.multi_hand_landmarks[0]
        lm = hand_landmarks.landmark

        fingers = []

        # Thumb: check if tip is to the right/left of base depending on handedness
        if lm[4].x > lm[3].x:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers (index, middle, ring, pinky)
        finger_tips = [8, 12, 16, 20]
        finger_bases = [6, 10, 14, 18]

        for tip, base in zip(finger_tips, finger_bases):
            if lm[tip].y < lm[base].y:
                fingers.append(1)
            else:
                fingers.append(0)

        total_fingers = fingers.count(1)
        return total_fingers

    def process_frame(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_result = self.pose.process(image_rgb)

        # Check for user visibility
        if pose_result.pose_landmarks:
            self.user_detected = True
            self.last_seen_time = time.time()
        elif time.time() - self.last_seen_time > 5:
            if self.user_detected:
                self.user_detected = False
                self.speak("Please stand in front of the camera")

        # If user is not detected
        if not pose_result.pose_landmarks and self.user_detected:
            self.speak("Adjust your position to be in front of the camera")

        # Draw pose landmarks
        if pose_result.pose_landmarks:
            self.mp_drawing.draw_landmarks(frame, pose_result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        # Hand gesture control
        finger_count = self.count_fingers(frame)
        if finger_count == 5 and not self.started:
            self.started = True
            self.session_start_time = time.time()
            self.speak("Pull-up session started")
        elif finger_count == 2 and self.started and not self.ended:
            self.ended = True
            self.speak(f"Session ended. You completed {self.counter} pull-ups")
            time.sleep(1)
            self.speak("Great job! Keep pushing your limits!")

        landmarks = pose_result.pose_landmarks.landmark if pose_result.pose_landmarks else None

        if landmarks:
            left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                             landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y]
            left_elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].x,
                          landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW].y]
            left_wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].x,
                          landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].y]

            angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)

            if self.started and not self.ended:
                if angle < self.pullup_threshold and not self.pull_up_complete:
                    self.pull_up_complete = True
                if angle > self.release_threshold and self.pull_up_complete:
                    self.pull_up_complete = False
                    self.update_counter()

        # Draw UI
        cv2.putText(frame, "Show 5 fingers to START, 2 fingers to END", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if self.started:
            elapsed_time = int(time.time() - self.session_start_time)
            cv2.putText(frame, f'Pull-ups: {self.counter}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.putText(frame, f'Time: {elapsed_time}s', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
            if self.paused:
                cv2.putText(frame, "PAUSED", (250, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

        return frame

def main():
    cap = cv2.VideoCapture(0)
    tracker = PullUpTracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = tracker.process_frame(frame)

        cv2.imshow("Pull-up Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
