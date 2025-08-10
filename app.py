import os
import sys
import subprocess
import time
from flask import Flask, jsonify, render_template

app = Flask(__name__)

EXERCISES = ['pushup', 'pullup', 'cardio', 'benchpress', 'dumbbells', 'lunges', 'shoulder', 'squat']

current_process = None
start_time = None
total_workout_time = 0  # Store total workout duration

@app.route('/')
def index():
    return render_template('user1.html', total_time=total_workout_time, calories=calculate_calories(total_workout_time))

@app.route('/start/<exercise>')
def start_exercise(exercise):
    global current_process, start_time, total_workout_time

    if exercise not in EXERCISES:
        return jsonify(success=False, error="Invalid exercise"), 400

    if current_process and current_process.poll() is None:
        stop_workout()

    start_time = time.time()
    return run_script(f"{exercise}.py")

@app.route('/stop')
def stop_workout():
    global current_process, start_time, total_workout_time

    if current_process and current_process.poll() is None:
        current_process.terminate()
        current_process.wait()

    if start_time:
        duration = int(time.time() - start_time)
        total_workout_time += duration
        start_time = None

    return jsonify(success=True, total_time=total_workout_time, calories=calculate_calories(total_workout_time)), 200

def run_script(script_name):
    global current_process
    try:
        script_path = os.path.join(os.getcwd(), script_name)

        if not os.path.exists(script_path):
            return jsonify(success=False, error=f"Script '{script_name}' not found."), 404

        current_process = subprocess.Popen(
            [sys.executable, script_path], 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True
        )

        return jsonify(success=True, message=f"Script '{script_name}' started."), 200

    except Exception as e:
        return jsonify(success=False, error=str(e)), 500

def calculate_calories(time_sec):
    return round(time_sec * 0.15)  # Approx: 1 min = 9 calories

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5012, debug=True)
