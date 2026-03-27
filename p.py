# Imports remain the same
import cv2
import face_recognition
import os
import serial
import time
import numpy as np
import threading
import csv
import datetime
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import math
import mediapipe as mp
import traceback

# --- Configuration --- (Ensure these are correct)
KNOWN_FACES_DIR = 'known_faces'
# *** CHANGE THIS to your Arduino's serial port ***
SERIAL_PORT = '/dev/cu.usbserial-10' # Example for macOS, change as needed
BAUD_RATE = 9600
WEBCAM_INDEX = 0
RECOGNITION_TOLERANCE = 0.6
LOG_FILE = 'study_log.csv'
GRAPH_FILE = 'daily_study_graph.png'

# --- Email Configuration ---
# *** URGENT: VERIFY CREDENTIALS & GOOGLE ACCOUNT SECURITY ***
EMAIL_SENDER = 'aldky8076@gmail.com'
EMAIL_PASSWORD = 'zyzgeewttjsajens' # 16-digit App Password
EMAIL_RECIPIENTS = ['deepak17943@gmail.com', 'dlsaa8076@gmail.com']
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587

# --- Gesture/Brightness Configuration --- (Unchanged)
PINCH_THRESHOLD = 0.06
BRIGHTNESS_SENSITIVITY = 1000
DEFAULT_BRIGHTNESS = 128
BRIGHTNESS_LOCKOUT_S = 5.0
RELEASE_CONFIRM_DURATION = 0.5

# --- MediaPipe Setup --- (Unchanged)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- Global Variables --- (Unchanged)
known_face_encodings = []; known_face_names = []; ser = None; stop_threads = False
current_user = None; session_id = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
arduino_user_present = False; arduino_is_dark = False; led_state_on = False
current_brightness = 0; brightness_lockout_end_time = 0.0
timer_seconds_initial = 0; timer_seconds_remaining = 0; timer_running = False
timer_paused = False; user_leave_time = None; is_pinching = False
last_pinch_y = 0.0; target_brightness_during_pinch = 0.0; potential_release_time = None

# --- Logging Function --- (Unchanged)
def log_event(event_type, user, duration_set=None, time_remaining=None, away_duration_sec=None, notes=''):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'); file_exists = os.path.isfile(LOG_FILE)
    try:
        with open(LOG_FILE, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Timestamp', 'SessionID', 'Event', 'User', 'DurationSetSec', 'TimeRemainingSec', 'AwayDurationSec', 'Notes']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists or os.path.getsize(LOG_FILE) == 0: writer.writeheader()
            log_data = { 'Timestamp': timestamp, 'SessionID': session_id, 'Event': event_type, 'User': user if user else 'N/A', 'DurationSetSec': duration_set if duration_set is not None else '', 'TimeRemainingSec': time_remaining if time_remaining is not None else '', 'AwayDurationSec': away_duration_sec if away_duration_sec is not None else '', 'Notes': notes }
            writer.writerow(log_data)
    except IOError as e: print(f"Error logging: {e}")
    except Exception as e: print(f"Unexpected error logging: {e}")

# --- Function Definition for Loading Faces --- (Unchanged)
def load_known_faces():
    global known_face_encodings, known_face_names; known_face_encodings = []; known_face_names = []
    print(f"Loading faces from '{KNOWN_FACES_DIR}'...")
    if not os.path.isdir(KNOWN_FACES_DIR): print(f"Error: Dir not found: '{os.path.abspath(KNOWN_FACES_DIR)}'"); return
    image_files_found = 0
    for name in os.listdir(KNOWN_FACES_DIR):
        if name.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(KNOWN_FACES_DIR, name); image_files_found += 1
            try:
                print(f" - Proc: {name}"); image = face_recognition.load_image_file(filepath); encodings = face_recognition.face_encodings(image)
                if encodings: known_face_encodings.append(encodings[0]); face_name = os.path.splitext(name)[0].replace("_", " ").title(); known_face_names.append(face_name); print(f"   - Loaded: {face_name}")
                else: print(f"   - Warn: No face in {name}")
            except Exception as e: print(f"   - Error: {filepath}: {e}")
        elif not name.startswith('.'): print(f" - Skip: {name}")
    if image_files_found == 0: print(f"Warn: No image files found.")
    print(f"Loaded {len(known_face_names)} known faces.")

# --- Function Definition for Serial Setup --- (Unchanged)
def setup_serial():
    global ser
    try:
        print(f"Connecting: {SERIAL_PORT} @ {BAUD_RATE} baud..."); ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print("Wait Arduino init..."); time.sleep(3)
        if ser.is_open: print("Serial connected."); ser.flushInput(); ser.flushOutput(); time.sleep(0.1); send_to_arduino("PYTHON_READY"); time.sleep(0.5); print("Ready signal sent."); return True
        else: print("Serial failed open."); ser = None; return False
    except serial.SerialException as e: print(f"\n*** Serial Error ***\n{e}\nCheck Port/Perms/Monitor."); ser = None; return False
    except Exception as e: print(f"Unexpected serial setup error: {e}"); ser = None; return False

# --- Reporting Functions ---
# ** ENHANCED VERSION WITH GRAPHING ** (Unchanged)
def generate_report_and_graph(log_filepath=LOG_FILE, graph_filepath=GRAPH_FILE):
    print(f"Generating report from {log_filepath}...")
    summary = f"Study Report for {datetime.date.today()}:\n"
    graph_generated_path = None
    try:
        if not os.path.exists(log_filepath) or os.path.getsize(log_filepath) == 0: summary += "\nLog empty/not found."; print("Log empty/not found."); return summary, None
        df = pd.read_csv(log_filepath)
        if df.empty: summary += "\nLog has no data."; print("Log empty."); return summary, None
        summary += f"\nLog contains {len(df)} entries."
        print(f"Log read with {len(df)} rows.")
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce'); df.dropna(subset=['Timestamp'], inplace=True)
        if df.empty: summary += "\nNo valid timestamps."; print("No valid timestamps."); return summary, None
        df['Date'] = df['Timestamp'].dt.date
        timer_set_events = df[df['Event'] == 'Timer Set'][['Date', 'User', 'DurationSetSec']].copy()
        away_events = df[df['Event'] == 'User Returned'][['Date', 'User', 'AwayDurationSec']].copy()
        timer_set_events['DurationSetSec'] = pd.to_numeric(timer_set_events['DurationSetSec'], errors='coerce').fillna(0)
        away_events['AwayDurationSec'] = pd.to_numeric(away_events['AwayDurationSec'], errors='coerce').fillna(0)
        daily_set_time = timer_set_events.groupby(['Date', 'User'])['DurationSetSec'].sum()
        daily_away_time = away_events.groupby(['Date', 'User'])['AwayDurationSec'].sum()
        daily_summary = pd.DataFrame(daily_set_time).join(daily_away_time, how='left').fillna(0)
        if not daily_summary.empty:
            daily_summary['ApproxStudySec'] = daily_summary['DurationSetSec'] - daily_summary['AwayDurationSec']
            daily_summary['ApproxStudySec'] = daily_summary['ApproxStudySec'].clip(lower=0)
            daily_summary['ApproxStudyHours'] = daily_summary['ApproxStudySec'] / 3600.0
            print("\nCalculated Daily Summary:\n", daily_summary)
            summary += "\n\nApproximate Daily Study Time (Hours):\n"
            for index, row in daily_summary.iterrows(): date, user = index; summary += f"- {date} | {user}: {row['ApproxStudyHours']:.2f} hrs\n"
            print(f"\nGenerating graph: {graph_filepath}")
            try:
                fig, ax = plt.subplots(figsize=(12, 7))
                plot_data = daily_summary.reset_index().pivot(index='Date', columns='User', values='ApproxStudyHours').fillna(0)
                if not plot_data.empty:
                     plot_data.plot(kind='bar', ax=ax, stacked=False, width=0.8); ax.set_title('Approximate Daily Study Hours per User'); ax.set_xlabel('Date'); ax.set_ylabel('Approximate Study Hours')
                     ax.tick_params(axis='x', rotation=45); ax.legend(title='User'); ax.grid(axis='y', linestyle='--'); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                     plt.tight_layout(); plt.savefig(graph_filepath); plt.close(fig); print(f"Graph saved: {graph_filepath}"); graph_generated_path = graph_filepath
                else: print("No data to plot."); summary += "\n(No data for graph)"
            except ImportError: print("Error: Matplotlib missing."); summary += "\n(Graph failed: Matplotlib missing)"
            except Exception as plot_e: print(f"Error plotting: {plot_e}"); summary += f"\n(Graph failed: {plot_e})"; traceback.print_exc()
        else: summary += "\nNo timer sessions found."; print("No timer data.")
    except FileNotFoundError: summary += f"\nLog not found: {log_filepath}."; print(f"Log not found: {log_filepath}.")
    except pd.errors.EmptyDataError: summary += f"\nLog empty: {log_filepath}."; print(f"Log empty: {log_filepath}.")
    except Exception as e: summary += f"\nError generating report: {e}"; print(f"Error generating report: {e}"); traceback.print_exc()
    return summary, graph_generated_path

# EMAIL FUNCTION WITH DEBUG PRINTS (Unchanged)
def send_email_report(subject, body, graph_filepath, recipients):
    print("\n--- Attempting to Send Email ---")
    if not EMAIL_SENDER or not EMAIL_PASSWORD or not recipients: print("Email config missing. Skipping."); return False
    if not isinstance(recipients, list) or not all(isinstance(email, str) for email in recipients): print(f"ERROR: Recipients not valid list: {recipients}"); return False
    if not recipients: print("No recipients. Skipping."); return False
    print(f"Sender: {EMAIL_SENDER}"); print(f"Recipients: {', '.join(recipients)}"); print(f"Subject: {subject}")
    message = MIMEMultipart(); message['From'] = EMAIL_SENDER; message['To'] = ", ".join(recipients); message['Subject'] = subject
    print("Attaching body...");
    try: message.attach(MIMEText(body, 'plain', 'utf-8')); print("Body attached.")
    except Exception as e: print(f"Error attaching body: {e}"); return False
    if graph_filepath and os.path.exists(graph_filepath):
        print(f"Attaching graph: {graph_filepath}")
        try:
            with open(graph_filepath, 'rb') as fp: img = MIMEImage(fp.read())
            img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(graph_filepath)); message.attach(img); print("Graph attached.")
        except FileNotFoundError: print(f"Error attaching graph: File not found: {graph_filepath}")
        except Exception as e: print(f"Error reading/attaching graph: {graph_filepath}: {e}")
    elif graph_filepath: print(f"Graph specified but not found: {graph_filepath}. Sending without.")
    else: print("No graph generated/specified. Sending without.")
    server = None
    try:
        print(f"Connecting: {SMTP_SERVER}:{SMTP_PORT}..."); server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=20)
        print("Connected. Enabling TLS..."); server.starttls(); print("TLS enabled. Logging in...")
        server.login(EMAIL_SENDER, EMAIL_PASSWORD); print("Login successful."); print("Sending...")
        server.sendmail(EMAIL_SENDER, recipients, message.as_string()); print(f"Email sent successfully."); print("--- Email Sending Complete ---"); return True
    except smtplib.SMTPAuthenticationError as e: print(f"ERROR: SMTP Authentication Error (CHECK APP PASSWORD / GOOGLE ACCOUNT SECURITY). Details: {e}"); return False
    except smtplib.SMTPServerDisconnected: print("ERROR: SMTP Server disconnected."); return False
    except smtplib.SMTPException as e: print(f"ERROR: SMTPException: {e}"); return False
    except TimeoutError: print(f"ERROR: SMTP connection timeout."); return False
    except Exception as e: print(f"ERROR: Unexpected email error: {e}"); traceback.print_exc(); return False
    finally:
        if server:
            try: print("Closing SMTP connection."); server.quit()
            except: print("Error closing SMTP connection.")
        print("--- Email Function Finished ---")


# --- Serial Communication --- (Unchanged)
def send_to_arduino(message):
    if ser and ser.is_open:
        try: ser.write((message + '\n').encode('utf-8')); ser.flush(); time.sleep(0.03)
        except serial.SerialTimeoutException: print(f"Warn: Serial write timeout.")
        except serial.SerialException as e: print(f"Error: Serial write: {e}")
        except Exception as e: print(f"Error: Serial send: {e}")

def read_from_arduino():
    global arduino_user_present, arduino_is_dark, stop_threads
    print("Serial reader thread started.")
    while not stop_threads:
        if ser and ser.is_open:
            try:
                if ser.in_waiting > 0:
                    message = ser.readline().decode('utf-8', errors='ignore').strip()
                    if not message: continue
                    if   message == "USER_PRESENT":      arduino_user_present = True
                    elif message == "USER_ABSENT":       arduino_user_present = False
                    elif message == "IS_DARK":           arduino_is_dark = True
                    elif message == "IS_LIGHT":          arduino_is_dark = False
                    elif message == "ACK_PYTHON_READY":  print("Arduino ACK.")
            except serial.SerialException as e: print(f"Error: Serial read: {e}. Pausing..."); time.sleep(3)
            except Exception as e: print(f"Error processing Arduino msg: {e}"); traceback.print_exc(); time.sleep(1)
        else:
            if not stop_threads: time.sleep(2)
        time.sleep(0.02)
    print("Serial reader thread finished.")


# --- Timer Function --- (Unchanged - Does NOT trigger email)
def run_timer():
    global timer_seconds_remaining, timer_running, timer_paused, stop_threads, current_user
    print("Timer thread started.")
    user_for_log = current_user
    log_event('Timer Started', user_for_log, duration_set=timer_seconds_initial, time_remaining=timer_seconds_remaining)
    send_to_arduino("TIMER_RUNNING_PY")
    last_lcd_update_time = 0; last_decrement_time = time.monotonic()
    while timer_seconds_remaining > 0 and not stop_threads:
        current_loop_time_monotonic = time.monotonic()
        if timer_running and not timer_paused:
            if current_loop_time_monotonic - last_lcd_update_time >= 1.0:
                 minutes = timer_seconds_remaining // 60; seconds = timer_seconds_remaining % 60; time_str = f"{minutes:02d}:{seconds:02d}"
                 send_to_arduino(f"LCD1:Time Left:"); send_to_arduino(f"LCD2:{time_str}"); last_lcd_update_time = current_loop_time_monotonic
            if current_loop_time_monotonic - last_decrement_time >= 1.0:
                  if timer_seconds_remaining > 0: timer_seconds_remaining -= 1
                  last_decrement_time += 1.0
            time.sleep(0.05)
        elif timer_paused:
             if current_loop_time_monotonic - last_lcd_update_time >= 2.0: send_to_arduino("LCD2: Paused"); last_lcd_update_time = current_loop_time_monotonic
             last_decrement_time = current_loop_time_monotonic; time.sleep(0.5)
        else: print("Timer thread: timer_running is false."); break
    if not stop_threads:
        if timer_seconds_remaining <= 0:
            print("Timer finished normally!")
            log_event('Timer Finished', user_for_log, duration_set=timer_seconds_initial, time_remaining=0)
            send_to_arduino("TIMER_DONE_PY"); send_to_arduino("LCD1:Time's Up!"); send_to_arduino("LCD2:")
            # No email trigger here
    else: print("Timer thread stopped externally."); log_event('Timer Interrupted', user_for_log, duration_set=timer_seconds_initial, time_remaining=timer_seconds_remaining); send_to_arduino("TIMER_STOPPED_PY")
    timer_running = False; timer_paused = False
    print("Timer thread finished.")


# --- Main Function ---
def main():
    global timer_seconds_initial, timer_seconds_remaining, timer_running, timer_paused
    global stop_threads, current_user, session_id
    global arduino_user_present, arduino_is_dark, led_state_on, current_brightness
    global brightness_lockout_end_time, is_pinching, last_pinch_y, target_brightness_during_pinch
    global potential_release_time
    global ser

    print(f"Starting Study Session Monitor - Session ID: {session_id}")
    print(f"Python Script Started: {datetime.datetime.now()}")
    log_event('Session Start', user='System')

    load_known_faces();
    if not known_face_names: print("Warning: No known faces loaded.")

    if not setup_serial(): log_event('Session End', user='System', notes='Serial failed.'); print("CRITICAL: Exiting."); return

    serial_thread = threading.Thread(target=read_from_arduino, name="SerialReader", daemon=True); serial_thread.start()
    print("Waiting for Arduino states..."); time.sleep(2.0); print(f"Initial States: UserPresent={arduino_user_present}, IsDark={arduino_is_dark}")

    print(f"Initializing webcam {WEBCAM_INDEX}..."); video_capture = cv2.VideoCapture(WEBCAM_INDEX); time.sleep(1.0)

    # *** CORRECTED Webcam Check ***
    if not video_capture.isOpened():
        print(f"CRITICAL Error: Could not open webcam {WEBCAM_INDEX}.")
        stop_threads = True
        log_event('Session End', user='System', notes='Webcam failed to open.')
        # Attempt to close serial port if it was opened
        if ser and ser.is_open:
            print("Closing serial port due to webcam failure...")
            ser.close()
        return # Exit main function if webcam fails

    print("Webcam initialized."); timer_thread = None; face_detection_active = True; user_name_prompted = False
    print("Starting main application loop..."); send_to_arduino("LCD1:Show Face"); send_to_arduino("LCD2:")

    # --- Main Application Loop ---
    while not stop_threads:
        ret, frame = video_capture.read()
        if not ret: print("Warn: Skip frame."); time.sleep(0.1); continue
        frame_height, frame_width, _ = frame.shape

        frame.flags.writeable = False; rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); frame.flags.writeable = True

        # --- Face Detection ---
        if face_detection_active and not user_name_prompted:
            # (Face detection logic unchanged)
            small_rgb_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)
            face_locations = face_recognition.face_locations(small_rgb_frame)
            face_encodings = face_recognition.face_encodings(small_rgb_frame, face_locations) if face_locations else []
            recognized_name = None
            if face_encodings and known_face_encodings:
                 for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=RECOGNITION_TOLERANCE)
                    if True in matches:
                         face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                         best_match_index = np.argmin(face_distances)
                         if matches[best_match_index]: recognized_name = known_face_names[best_match_index]; break
            if recognized_name:
                current_user = recognized_name; user_name_prompted = True; print(f"Face detected: {current_user}"); log_event('User Recognized', user=current_user)
                send_to_arduino(f"LCD1:Hi {current_user}!"); send_to_arduino(f"LCD2:Set timer (mins)")
                timer_minutes = -1
                while timer_minutes < 0:
                    try: timer_input = input(f" Enter timer (mins) or 0 to skip: "); timer_minutes = int(timer_input);
                    except ValueError: print("Invalid input.")
                if timer_minutes > 0:
                    timer_seconds_initial = timer_minutes * 60; timer_seconds_remaining = timer_seconds_initial; log_event('Timer Set', user=current_user, duration_set=timer_seconds_initial)
                    timer_running = True; timer_paused = False; timer_thread = threading.Thread(target=run_timer, name="TimerThread", daemon=True); timer_thread.start()
                    print(f"Timer set for {timer_minutes} mins."); send_to_arduino("LCD1:Timer Running"); send_to_arduino("LCD2:")
                else: print("Timer skipped."); send_to_arduino("LCD1:Gesture Control"); send_to_arduino("LCD2:Active")
                face_detection_active = False; print("-> Gesture control phase.")

        # --- Check Timer Pause/Resume based on Arduino state ---
        if timer_running:
            # *** CORRECTED Timer Resume Logic ***
            if not arduino_user_present and not timer_paused:
                 print("Main Loop: User absent -> Pausing timer.")
                 timer_paused = True; user_leave_time = datetime.datetime.now(); log_event('User Left', current_user, time_remaining=timer_seconds_remaining)
                 send_to_arduino("TIMER_PAUSED_PY")
            elif arduino_user_present and timer_paused:
                 print("Main Loop: User present -> Resuming timer.")
                 timer_paused = False
                 if user_leave_time:
                     # Use try-except block for calculation and logging
                     try:
                         away_duration_sec = int((datetime.datetime.now() - user_leave_time).total_seconds())
                         log_event('User Returned', current_user, time_remaining=timer_seconds_remaining, away_duration_sec=away_duration_sec)
                     except Exception as time_e:
                         print(f"Error calculating away duration: {time_e}")
                 user_leave_time = None # Reset regardless of calculation success
                 send_to_arduino("TIMER_RUNNING_PY") # Notify Arduino timer resumed


        # --- LED ON/OFF Logic ---
        if not face_detection_active:
            should_led_be_on = arduino_user_present and arduino_is_dark
            if should_led_be_on != led_state_on:
                if should_led_be_on: print("Main Loop: Conditions met -> LED ON"); current_brightness = DEFAULT_BRIGHTNESS; send_to_arduino(f"SET_BRIGHTNESS:{current_brightness}"); led_state_on = True; is_pinching = False; potential_release_time = None
                else: print("Main Loop: Conditions NOT met -> LED OFF"); current_brightness = 0; send_to_arduino("SET_BRIGHTNESS:0"); led_state_on = False; is_pinching = False; potential_release_time = None

        # --- Hand Tracking & Gesture Control Phase ---
        if led_state_on:
            # (Gesture detection and state machine logic - unchanged)
            results = hands.process(rgb_frame)
            pinch_detected_this_frame = False; current_pinch_y_norm = 0.0; calculated_distance = -1.0; hand_landmarks_for_drawing = None
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_landmarks_for_drawing = hand_landmarks
                    try:
                        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]; index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        calculated_distance = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
                        if calculated_distance < PINCH_THRESHOLD: pinch_detected_this_frame = True; current_pinch_y_norm = (thumb_tip.y + index_tip.y) / 2.0; break
                    except IndexError: print("Warn: Landmark index error."); continue
                    except Exception as lm_e: print(f"Error processing landmarks: {lm_e}"); continue
            can_control_brightness = time.time() >= brightness_lockout_end_time
            # State: Start Pinch
            if not is_pinching and pinch_detected_this_frame and can_control_brightness: print(f"DEBUG: Pinch STARTED (Dist: {calculated_distance:.3f})"); is_pinching = True; potential_release_time = None; last_pinch_y = current_pinch_y_norm; target_brightness_during_pinch = float(current_brightness)
            # State: Continue Pinch
            elif is_pinching and pinch_detected_this_frame and can_control_brightness:
                potential_release_time = None; delta_y_norm = last_pinch_y - current_pinch_y_norm; brightness_change = delta_y_norm * BRIGHTNESS_SENSITIVITY
                if abs(brightness_change) > 0.5 :
                    target_brightness_during_pinch += brightness_change; target_brightness_during_pinch = max(0.0, min(255.0, target_brightness_during_pinch)); new_brightness_int = int(round(target_brightness_during_pinch))
                    if new_brightness_int != current_brightness: current_brightness = new_brightness_int; send_to_arduino(f"SET_BRIGHTNESS:{current_brightness}")
                last_pinch_y = current_pinch_y_norm
            # State: Potential Release
            elif is_pinching and not pinch_detected_this_frame and can_control_brightness:
                if potential_release_time is None: potential_release_time = time.time()
                elif (time.time() - potential_release_time) > RELEASE_CONFIRM_DURATION: print(f"DEBUG: Pinch RELEASE CONFIRMED - Final Brightness: {current_brightness}"); is_pinching = False; potential_release_time = None; brightness_lockout_end_time = time.time() + BRIGHTNESS_LOCKOUT_S; print(f"DEBUG: Brightness locked out for {BRIGHTNESS_LOCKOUT_S:.1f}s.")
            # State: Reset potential release if pinch resumes or during lockout
            elif is_pinching and pinch_detected_this_frame: potential_release_time = None
            elif is_pinching and not pinch_detected_this_frame and not can_control_brightness: potential_release_time = None
            # Draw Landmarks & Status Text
            if frame.flags.writeable:
                 if hand_landmarks_for_drawing: mp_drawing.draw_landmarks(frame, hand_landmarks_for_drawing, mp_hands.HAND_CONNECTIONS)
                 if is_pinching and pinch_detected_this_frame: thumb_tip = hand_landmarks_for_drawing.landmark[mp_hands.HandLandmark.THUMB_TIP]; index_tip = hand_landmarks_for_drawing.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]; thumb_px=(int(thumb_tip.x*frame_width), int(thumb_tip.y*frame_height)); index_px=(int(index_tip.x*frame_width), int(index_tip.y*frame_height)); cv2.line(frame, thumb_px, index_px, (0, 255, 0), 3)
                 status_text = f"BRT: {current_brightness}"; lockout_remaining = max(0.0, brightness_lockout_end_time - time.time())
                 if lockout_remaining > 0: status_text += f" Locked: {lockout_remaining:.1f}s"
                 elif is_pinching: status_text += " (Adj)"
                 elif potential_release_time is not None: status_text += " (Release?)"
                 dist_text = f"PinchDist: {calculated_distance:.3f}" if calculated_distance >= 0 else "PinchDist: N/A"
                 cv2.putText(frame, status_text, (15, frame_height - 35), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
                 cv2.putText(frame, dist_text, (15, frame_height - 15), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)


        # --- Display Frame ---
        cv2.imshow('StudyCam Feed - Press Q to Quit', frame)

        # --- Exit Condition Check ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exit requested ('q' pressed).")
            stop_threads = True # Signal threads to stop
            break # Exit main loop

    # --- // End of Main Loop // ---

    # --- *** REPORTING MOVED HERE - Executes when loop ends (e.g., 'q' pressed) *** ---
    print("\nMain loop finished.")
    run_reporting = True # Set to False to disable reporting/email on exit
    if run_reporting:
        print("Generating final report on exit...")
        # Ensure generate_report_and_graph is defined correctly above
        summary_text, generated_graph_path = generate_report_and_graph(LOG_FILE, GRAPH_FILE)
        print("\n--- Final Report Summary ---")
        print(summary_text)
        print("--- End Report ---")

        # Send email only if recipients are configured
        if EMAIL_RECIPIENTS:
            print("\nAttempting to send final email report...")
            subject = f"Study Session Report (Exited) - {datetime.date.today()}"
            # Ensure send_email_report is defined correctly above
            send_email_report(subject, summary_text, generated_graph_path, EMAIL_RECIPIENTS)
        else:
            print("\nEmail recipients not configured. Skipping email on exit.")
    else:
         print("Reporting disabled on exit.")
    # --- *** END OF REPORTING SECTION *** ---


    # --- Cleanup ---
    print("\nCleaning up resources...")
    # stop_threads = True # Ensure flag is set (already set if 'q' pressed)
    if timer_thread and timer_thread.is_alive(): print("Waiting for timer thread..."); timer_thread.join(timeout=1.5)
    print("Turning LED off..."); send_to_arduino("SET_BRIGHTNESS:0"); time.sleep(0.1)
    if video_capture.isOpened(): video_capture.release(); print("Webcam released.")
    cv2.destroyAllWindows();
    for i in range(5): cv2.waitKey(1); # print("OpenCV windows closed.") # Reduce noise
    log_event('Session End', user=current_user if current_user else 'System', notes='Program exited.')
    print(f"Study Session Monitor Finished: {datetime.datetime.now()}")


# --- Main Execution Block --- (Unchanged)
if __name__ == '__main__':
    serial_needs_close = False
    try: main()
    except KeyboardInterrupt: print("\nProgram interrupted (Ctrl+C)."); stop_threads = True; log_event('Session End', user='System', notes='Interrupted by user (Ctrl+C)'); serial_needs_close = True
    except Exception as e: print(f"\nCRITICAL UNEXPECTED ERROR in main: {e}"); traceback.print_exc(); stop_threads = True; log_event('Session End', user='System', notes=f'Crashed with error: {e}'); serial_needs_close = True
    finally:
        print("\nExecuting final cleanup...");
        if 'ser' in globals() and ser and (ser.is_open or serial_needs_close):
            print("Closing serial port...");
            try:
                 if ser.is_open: send_to_arduino("SET_BRIGHTNESS:0"); time.sleep(0.1); ser.flush(); ser.close(); print("Serial port closed.")
                 else: print("Serial port was already closed.")
            except Exception as e: print(f"Error during final serial cleanup: {e}")
        else: print("Serial port was not open or not initialized.")
        try:
             if 'hands' in globals() and hands: hands.close(); print("MediaPipe Hands closed.")
        except Exception as e: print(f"Error closing MediaPipe Hands: {e}")
        print("Cleanup complete. Exiting.")