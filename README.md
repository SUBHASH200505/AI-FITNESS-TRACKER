# 🏋️‍♂️ AI Fitness Tracker

**AI-powered personal fitness assistant** that tracks exercises using computer vision, corrects posture in real time, calculates calories burned, suggests diet plans, and keeps you motivated with music — all tailored to your fitness goals.

---

## 📌 Features

- **🎯 Gesture-based Exercise Tracking**  
  Uses computer vision to detect and recognize workout movements such as squats, push-ups, and jumping jacks.

- **✅ Real-time Pose Correction**  
  Provides instant feedback to help maintain proper form and reduce injury risk.

- **🥗 Personalized Diet Plans**  
  Suggests diet recommendations based on user goals — weight loss, weight gain, or muscle building.

- **🎵 Motivational Music**  
  Plays music during workouts to keep energy and motivation high.

- **📊 Progress Tracking**  
  Displays workout history, completed sets, and performance trends over time.

- **🔥 Calorie Burn Estimation**  
  Calculates calories burned based on exercise type, duration, and intensity.

- **💪 Customized Workout Plans**  
  Generates exercise routines tailored to user fitness levels and goals.

---

## 🛠 Tech Stack

- **Programming Language:** Python  
- **Computer Vision:** OpenCV, Mediapipe  
- **Machine Learning:** TensorFlow / PyTorch (for pose classification)  
- **UI/UX:** Tkinter / Flask (for interactive interface)  
- **Audio:** Pygame / Playsound (for music playback)  
- **Data Handling:** Pandas, NumPy  

---

## 🚀 How It Works

1. **Pose Detection**  
   Uses Mediapipe Pose/BlazePose to detect key body landmarks from the webcam feed.

2. **Exercise Recognition**  
   Classifies detected poses into exercise types and counts repetitions.

3. **Form Correction**  
   Compares live pose coordinates with ideal form data and provides feedback.

4. **Calories & Progress**  
   Tracks exercise duration, calculates calories burned, and logs progress.

5. **Goal-Oriented Plans**  
   Asks the user to choose a goal (weight loss, weight gain, endurance, etc.) and suggests exercises & diet.

---

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-fitness-tracker.git
   cd ai-fitness-tracker
pip install -r requirements.txt
