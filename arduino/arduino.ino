#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// --- Configuration ---
LiquidCrystal_I2C lcd(0x27, 16, 2); // Address, Columns, Rows (Check your address)

// Sensor/Actuator Pins (User Specified)
const int trigPin = 8;        // Ultrasonic Trig pin
const int echoPin = 9;        // Ultrasonic Echo pin
const int buzzerPin = 10;       // Buzzer pin
const int ldrDigitalPin = 13;     // Digital pin connected to LDR module's D0 output
const int ledStripPin = 11;       // <<< MUST BE a PWM pin (e.g., ~3, ~5, ~6, ~9, ~10, ~11 on Uno/Nano)

// Default Brightness when Python is not connected
const int DEFAULT_BRIGHTNESS_NO_PYTHON = 128; // Brightness 0-255

// Thresholds and Timings
const long PRESENCE_CHECK_INTERVAL = 500; // ms
const float MAX_DISTANCE_CM = 60.0;     // cm
const long LDR_CHECK_INTERVAL = 1000; // ms
unsigned long PAUSE_TIMEOUT_WARN = 10000; // 10s for buzzer warning
unsigned long PAUSE_TIMEOUT_ALERT = 25000; // 25s for buzzer alert

// --- Global Variables ---
bool lastUserPresentState = false;
bool lastIsDarkState = false;
unsigned long lastPresenceCheckTime = 0;
unsigned long lastLdrCheckTime = 0;
bool pythonReady = false;
bool timerRunningArduino = false;
unsigned long pauseStartTime = 0;
bool buzzerOn = false;
int currentLedBrightness = 0;

// --- Setup ---
void setup() {
  Serial.begin(9600);
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  pinMode(buzzerPin, OUTPUT);
  pinMode(ledStripPin, OUTPUT);
  pinMode(ldrDigitalPin, INPUT);

  digitalWrite(buzzerPin, LOW); // Ensure buzzer is off
  analogWrite(ledStripPin, 0);  // Start with LEDs OFF
  currentLedBrightness = 0;

  lcd.init();
  lcd.backlight();
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("System Booting");
  lcd.setCursor(0, 1);
  lcd.print("Waiting Python..");

  checkUserPresence(false);
  checkLight(false);
  checkSetLedState(); // Set initial LED based on conditions
}

// --- Main Loop ---
void loop() {
  checkSerial();
  checkUserPresence(false);
  checkLight(false);
  checkSetLedState();
  handleBuzzerAlert();
}

// --- Helper Functions ---

void checkSetLedState() {
    bool conditions_met = lastUserPresentState && lastIsDarkState;
    if (conditions_met) {
        if (!pythonReady) { // Python not connected
            if (currentLedBrightness == 0) { // Only turn on if currently off
                 analogWrite(ledStripPin, DEFAULT_BRIGHTNESS_NO_PYTHON);
                 currentLedBrightness = DEFAULT_BRIGHTNESS_NO_PYTHON;
            }
        } // If Python is ready, Python controls brightness via commands
    } else { // Conditions not met
        if (currentLedBrightness != 0) { // Only turn off if currently on
            analogWrite(ledStripPin, 0);
            currentLedBrightness = 0;
        }
    }
}

void checkUserPresence(bool forceReport) {
  unsigned long currentTime = millis();
  if (forceReport || (currentTime - lastPresenceCheckTime >= PRESENCE_CHECK_INTERVAL)) {
    lastPresenceCheckTime = currentTime;
    // Ultrasonic reading
    digitalWrite(trigPin, LOW); delayMicroseconds(2);
    digitalWrite(trigPin, HIGH); delayMicroseconds(10);
    digitalWrite(trigPin, LOW);
    long duration = pulseIn(echoPin, HIGH, (long)(MAX_DISTANCE_CM * 2 * 10000 / 34) + 6000 );
    float distance = (duration * 0.0343) / 2.0;
    bool currentUserPresent = (distance > 2 && distance < MAX_DISTANCE_CM);

    if (forceReport || (currentUserPresent != lastUserPresentState)) {
      bool previousPresenceState = lastUserPresentState; // Store state before update
      lastUserPresentState = currentUserPresent; // Update internal state

      if (currentUserPresent) { // User JUST returned or is present initially
        Serial.println("USER_PRESENT");
        pauseStartTime = 0; // Reset pause timer
        if (previousPresenceState == false) { // Only turn off buzzer if they JUST returned
             buzzerOn = false;
             digitalWrite(buzzerPin, LOW); // *** Explicitly turn buzzer OFF here ***
        }
      } else { // User JUST left or is absent initially
        Serial.println("USER_ABSENT");
        if(timerRunningArduino) { // Start pause timer only if timer was running
             pauseStartTime = millis();
        }
      }
    }
  }
}


void checkLight(bool forceReport) {
   unsigned long currentTime = millis();
   if (forceReport || (currentTime - lastLdrCheckTime >= LDR_CHECK_INTERVAL)) {
      lastLdrCheckTime = currentTime;
      int lightState = digitalRead(ldrDigitalPin);
      // Verify your LDR module: HIGH means DARK? Or LOW means DARK?
      bool currentIsDark = (lightState == HIGH); // Assumes HIGH=DARK

      if (forceReport || (currentIsDark != lastIsDarkState)) {
         lastIsDarkState = currentIsDark; // Update state
         Serial.println(currentIsDark ? "IS_DARK" : "IS_LIGHT"); // Report change
      }
   }
}

void checkSerial() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();

    if (command == "PYTHON_READY") {
         if (!pythonReady) {
             pythonReady = true;
             Serial.println("ACK_PYTHON_READY");
             lcd.clear(); lcd.print("Python Ready"); lcd.setCursor(0,1); lcd.print("Waiting Cmds...");
             checkUserPresence(true); // Force report current states
             checkLight(true);
         }
    }
    else if (pythonReady) { // Process commands only if Python is ready
        if (command.startsWith("SET_BRIGHTNESS:")) {
          int brightness = command.substring(15).toInt();
          brightness = constrain(brightness, 0, 255);
          analogWrite(ledStripPin, brightness);
          currentLedBrightness = brightness;
        }
        else if (command == "TIMER_RUNNING_PY") {
             timerRunningArduino = true; pauseStartTime = 0; buzzerOn = false;
             digitalWrite(buzzerPin, LOW); // Ensure buzzer off when timer starts/resumes
             lcd.clear(); lcd.setCursor(0,0); lcd.print("Timer Running"); lcd.setCursor(0,1); lcd.print("");
        } else if (command == "TIMER_PAUSED_PY") {
             // Python handles timer logic pause, Arduino just shows message maybe
             lcd.setCursor(0,1); lcd.print("Paused");
        } else if (command == "TIMER_DONE_PY" || command == "TIMER_STOPPED_PY") {
             timerRunningArduino = false; buzzerOn = false;
             digitalWrite(buzzerPin, LOW); // Ensure buzzer off when timer stops/ends
             pauseStartTime = 0;
             if(command == "TIMER_DONE_PY") { lcd.clear(); lcd.setCursor(0,0); lcd.print("Timer Done!"); }
             else { lcd.clear(); lcd.setCursor(0,0); lcd.print("Timer Stopped"); }
             lcd.setCursor(0,1); lcd.print("");
        }
        else if (command.startsWith("LCD1:")) {
            lcd.setCursor(0,0); lcd.print("                "); lcd.setCursor(0,0); lcd.print(command.substring(5));
        } else if (command.startsWith("LCD2:")) {
             lcd.setCursor(0,1); lcd.print("                "); lcd.setCursor(0,1); lcd.print(command.substring(5));
        }
    }
  }
}

void handleBuzzerAlert() {
    if (timerRunningArduino && !lastUserPresentState && pauseStartTime > 0) {
        unsigned long timeAway = millis() - pauseStartTime;
        bool shouldBuzz = (timeAway >= PAUSE_TIMEOUT_ALERT);
        if (shouldBuzz && !buzzerOn) { // Turn ON only if threshold reached AND not already on
            buzzerOn = true;
            digitalWrite(buzzerPin, HIGH);
            lcd.setCursor(0,1); lcd.print("!!! ALERT !!! "); // Show alert message
        } else if (!shouldBuzz && buzzerOn) {
            // This case should ideally be handled by user return or timer stop commands,
            // but as a failsafe, turn off if alert time hasn't been reached but buzzer was somehow on.
            // buzzerOn = false; // Let user return logic handle this primarily
            // digitalWrite(buzzerPin, LOW);
        }
        // Warning message (optional, might overwrite timer)
        // else if (!buzzerOn && timeAway >= PAUSE_TIMEOUT_WARN) {
        //     lcd.setCursor(0,1); lcd.print("Please Come Back");
        // }
    } else { // Conditions for alert not met (timer not running, user present, or pause not started)
        if (buzzerOn) { // If buzzer was on, turn it off
             buzzerOn = false;
             digitalWrite(buzzerPin, LOW);
        }
    }
}

// Format Time function (not used by Arduino directly now)
String formatTime(long totalSeconds) { /* ... remains the same ... */ }
