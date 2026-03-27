# Smart Study Session Monitor

This is a comprehensive, AI-powered patent project designed to monitor, log, and enhance study sessions. It integrates hardware (Arduino sensors and LEDs) with a sophisticated Python backend using OpenCV, face recognition, and hand-gesture tracking.

## Features

- **Automated Face Recognition:** Identifies the user via webcam against known profiles (in `known_faces/`), personalizing the session setup.
- **Smart Timer & Presence Detection:** The user inputs a study timer. If the Arduino's ultrasonic/PIR sensor detects the user has left the desk, the timer automatically pauses, and resumes when they return.
- **Intelligent Lighting Control:** Integrates with an Arduino light sensor to automatically turn on a desk LED when the room is dark and the user is present.
- **Gesture-Controlled Brightness:** Uses MediaPipe hand tracking to allow the user to adjust the LED brightness simply by pinching their thumb and index finger and moving their hand up or down.
- **Activity Logging & Analytics:** Generates daily study reports (saving a daily graph using Matplotlib) and logs every event in a comprehensive CSV (`study_log.csv`).
- **Automated Email Reports:** Emails the daily study summary and graph directly to the user when the session ends.

## Prerequisites

### Hardware Requirements
- Arduino board connected via Serial (USB). Note: Update `SERIAL_PORT` in the code to match your specific setup (e.g., `/dev/cu.usbserial-10`).
- Proximity sensor (Ultrasonic/PIR) connected to the Arduino.
- Light sensor (LDR) connected to the Arduino.
- Dimmable LED component connected to the Arduino.
- Webcam for face and gesture recognition.

### Software Requirements
- Python 3.x
- Arduino IDE (for flashing the Arduino component, not included in this repo)
- C/C++ compiler tools (for `face_recognition` dependencies like `dlib`)

### Required Python Libraries
```bash
pip install opencv-python face_recognition pyserial numpy pandas matplotlib mediapipe
```

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd <repository-directory>
   ```

2. **Add Known Faces:**
   - Create a directory named `known_faces/` in the project root if it doesn't exist.
   - Place images of the users who will use the system in this folder. Name the files with the person's name (e.g., `john_doe.jpg`).

3. **Configure the Script (`p.py`):**
   - Update `SERIAL_PORT` to match your Arduino's serial port.
   - Update `EMAIL_SENDER`, `EMAIL_PASSWORD` (use an app password), and `EMAIL_RECIPIENTS` to configure the reporting feature.

4. **Run the Application:**
   ```bash
   python p.py
   ```

## Usage Instructions

1. **Start the Script:** Run `python p.py`. Ensure your Arduino is connected and your webcam is active.
2. **Face Identification:** Look into the webcam. The system will greet you by name if recognized and prompt you to set a study timer in minutes.
3. **Session Monitoring:**
   - **Leave tracking:** If you walk away, the Arduino detects your absence and pauses the timer. It resumes upon your return.
   - **Auto-Lighting:** If the room gets dark, the LED will automatically turn on.
4. **Gesture Controls:**
   - When the LED is on, hold your thumb and index finger close together (pinch) in front of the camera.
   - Move your pinched fingers up to increase brightness, or down to decrease it.
   - Release the pinch to lock in the brightness.
5. **End Session:**
   - Press the `q` key on the active OpenCV video window to exit.
   - The system will safely wind down, turn off the LED, save your session log/graph, and send an email report to the configured addresses.

## Output Files

- `study_log.csv`: A continuous log of all your sessions, pauses, and returns.
- `daily_study_graph.png`: A bar chart tracking approximate daily study hours, generated per user per day.

## Disclaimer

This is a patent-pending project showcasing the integration of physical sensor networks with computer vision and machine learning for smart environment optimization.
