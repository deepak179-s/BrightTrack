# BrightTrack: Integrated Smart Study Lamp System

**BrightTrack** is an AI-powered, IoT-enabled smart study lamp system designed to enhance student focus, monitor study habits, and provide adaptive lighting. It integrates an Arduino-based hardware network (Ultrasonic, LDR, MPU6050, ZX8020) with a sophisticated Python computer vision backend (OpenCV, MediaPipe, Face Recognition).

This project acts as a complete automated monitor for study sessions, providing real-time adaptive lighting, multimodal interaction, automated monitoring, and independent manual control.

## 🌟 How This Helps Students (The Problem It Solves)

Students engaged in prolonged study sessions often experience diminished focus or drowsiness without objective feedback mechanisms, significantly reducing learning efficiency. Furthermore, traditional lamps do not adapt to ambient light, require manual adjustment, and do not track study metrics to provide parents with visibility into study habits. 

**BrightTrack** solves these problems by:
1. **Preventing Dozing Off:** The integrated MPU6050 IMU detects sustained periods of inactivity (suggesting sleep) and provides an immediate auditory buzzer alert to prompt the student to regain alertness or take a break.
2. **Eliminating Study Interruptions:** The MediaPipe computer-vision capability allows students to intuitively control brightness simply by pinching their fingers mid-air, avoiding disruptive manual tweaks.
3. **Encouraging Adherence to Schedules:** The ultrasonic sensor automatically pauses the study timer the moment a student leaves their desk and alerts them if the absence is prolonged, improving time management.
4. **Providing Data-Driven Insights:** All study times, break intervals, and alerts are meticulously logged and automatically formatted into daily graphs and performance summaries, which are emailed to parents or guardians.

## 🛠️ Key Features & How It Works

- **Automated Face Recognition:** Identifies the user via webcam and greets them by name on the LCD display. By identifying the specific user from a database, it ensures logging accurately matches the person sitting down.
- **Smart Timer & Presence Detection:** A study timer pauses automatically if the user leaves the desk (detected via an HC-SR04 Ultrasonic sensor) and alerts on extended absence.
- **Adaptive Lighting Control:** Integrates an LDR (Light Dependent Resistor) to automatically switch the LED desk lamp on in low light (when the user is present) and off when ambient light is sufficient.
- **Gesture-Controlled Brightness:** Uses MediaPipe hand tracking so users can adjust LED brightness completely hands-free by pinching and moving their fingers.
- **Activity & Sleep Monitoring (MPU6050):** Uses an accelerometer/gyroscope to track physical activity. Prolonged inactivity triggers a buzzer alert to prevent the student from falling asleep.
- **Independent Manual Touch Control:** Features a ZX8020 touch sensor for manual on/off and brightness cycling (100 -> 180 -> 255 -> Off), ensuring fundamental lighting works independently of the AI system.
- **Activity Logging & Analytics:** Generates daily study reports (saving a daily graph using pandas and matplotlib) and logs every session event into `study_log.csv`.
- **Automated Email Reports:** Automatically compiles and emails the daily study summary and generated graphs to parents or guardians.

## 💻 Technical Hardware & Software Requirements

- **Microcontroller:** Arduino Uno (or compatible)
- **Sensors & Outputs:**
  - HC-SR04 Ultrasonic Sensor (Presence detection)
  - LDR / Photoresistor (Ambient light sensing)
  - MPU6050 Accelerometer/Gyroscope (Activity/Sleep monitoring)
  - ZX8020 Touch IC (Independent manual parallel control)
  - 12V Dimmable LED Strip, 1602 I2C LCD Display, Piezo Buzzer
- **Computer/Vision:** PC with USB Webcam.
- **Software:** Python 3.x, Arduino IDE
- **Python Libraries:** `opencv-python`, `face_recognition`, `pyserial`, `numpy`, `pandas`, `matplotlib`, `mediapipe`

## 🚀 Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/deepak179-s/BrightTrack.git
   cd BrightTrack
   ```

2. **Install Python Dependencies:**
   ```bash
   pip install opencv-python face_recognition pyserial numpy pandas matplotlib mediapipe
   ```

3. **Add Known Faces:**
   - Place images of the users who will use the system in the `known_faces/` directory.
   - Name the file exactly as you want the greeting to appear (e.g., `Deepak.jpg`).

4. **Configure the Script (`p.py`):**
   - Update `SERIAL_PORT` to match your Arduino's serial port (e.g., `/dev/cu.usbserial-10`).
   - Update `EMAIL_SENDER`, `EMAIL_PASSWORD` (use an App Password), and `EMAIL_RECIPIENTS` to configure the automated email reporting to parents.

## 📖 Usage Instructions

1. **Start Up:** Ensure the Arduino is connected and webcam is active, then run:
   ```bash
   python p.py
   ```
2. **Face Identification:** Look at the webcam. The LCD will greet you and prompt you to set a study timer (in minutes) via the terminal.
3. **Session Monitoring:**
   - Walk away, and the system pauses your timer and eventually triggers an alert.
   - Sit still or fall asleep, and the MPU6050 triggers a wake-up buzzer.
   - If the room gets dark, the LED automatically turns on.
4. **Gesture Controls:**
   - Pinch your thumb and index finger in view of the camera.
   - Move your hand up/down to adjust the LED brightness. Release the pinch to lock the brightness.
5. **End Session:**
   - Press the `q` key on the OpenCV video window to exit.
   - The system turns off the LED, generates a study graph, and emails the complete report.

## 📝 License / Disclaimer

This repository contains the software systems for the **BrightTrack** patent project. All components and integration methodologies are part of an Invention Disclosure framework.
