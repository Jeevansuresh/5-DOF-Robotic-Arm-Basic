// Robot arm - serial servo controller for ESP32 (NodeMCU ESP-32S)
// Board: "ESP32 Dev Module"   Library: ESP32Servo (Kevin Harrington)
//
// Wiring (signal pins):
//   Base -> P13   Shoulder -> P26   Elbow -> P14
//   Wrist -> P27  Gripper -> P15
//
// Protocol (115200 baud, one command per line):
//   B90  -> base to 90       S120 -> shoulder to 120
//   E45  -> elbow to 45      W90  -> wrist to 90
//   C30  -> gripper (claw) to 30
//   H    -> all joints home  ?    -> report current angles
// Servos glide toward the target (no sudden jumps -> less current spike).

#include <ESP32Servo.h>

const int NUM_JOINTS = 5;
const char  JOINT_ID[NUM_JOINTS]   = {'B',    'S',        'E',     'W',     'C'};
const char* JOINT_NAME[NUM_JOINTS] = {"Base", "Shoulder", "Elbow", "Wrist", "Gripper"};
const int   JOINT_PIN[NUM_JOINTS]  = {13,     26,         14,      27,      15};

// Tune these once you see your arm's real safe range
const int ANGLE_MIN[NUM_JOINTS]  = {0,   15,  0,   0,   20};
const int ANGLE_MAX[NUM_JOINTS]  = {180, 165, 180, 180, 160};
const int ANGLE_HOME[NUM_JOINTS] = {90,  90,  90,  90,  90};

const int STEP_DEG = 1;           // degrees moved per tick
const unsigned long TICK_MS = 12; // smaller = faster motion

Servo servos[NUM_JOINTS];
int currentAngle[NUM_JOINTS];
int targetAngle[NUM_JOINTS];

String inputLine = "";
unsigned long lastTick = 0;

int findJoint(char id) {
  for (int i = 0; i < NUM_JOINTS; i++) if (JOINT_ID[i] == id) return i;
  return -1;
}

void reportAngles() {
  Serial.print("POS");
  for (int i = 0; i < NUM_JOINTS; i++) {
    Serial.print(' '); Serial.print(JOINT_ID[i]); Serial.print(targetAngle[i]);
  }
  Serial.println();
}

void handleCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;
  char id = toupper(cmd.charAt(0));

  if (id == 'H') {
    for (int i = 0; i < NUM_JOINTS; i++) targetAngle[i] = ANGLE_HOME[i];
    Serial.println("OK HOME");
    return;
  }
  if (id == '?') { reportAngles(); return; }

  int j = findJoint(id);
  if (j < 0 || cmd.length() < 2) { Serial.print("ERR "); Serial.println(cmd); return; }

  int angle = constrain(cmd.substring(1).toInt(), ANGLE_MIN[j], ANGLE_MAX[j]);
  targetAngle[j] = angle;
  Serial.print("OK "); Serial.print(JOINT_NAME[j]); Serial.print(' '); Serial.println(angle);
}

void setup() {
  Serial.begin(115200);

  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);

  for (int i = 0; i < NUM_JOINTS; i++) {
    servos[i].setPeriodHertz(50);
    servos[i].attach(JOINT_PIN[i], 500, 2400);
    currentAngle[i] = targetAngle[i] = ANGLE_HOME[i];
    servos[i].write(ANGLE_HOME[i]);
    delay(250);  // stagger startup so all servos don't pull current at once
  }
  Serial.println("READY");
  reportAngles();
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (inputLine.length()) handleCommand(inputLine);
      inputLine = "";
    } else if (inputLine.length() < 16) {
      inputLine += c;
    }
  }

  if (millis() - lastTick >= TICK_MS) {
    lastTick = millis();
    for (int i = 0; i < NUM_JOINTS; i++) {
      if (currentAngle[i] < targetAngle[i])
        currentAngle[i] = min(currentAngle[i] + STEP_DEG, targetAngle[i]);
      else if (currentAngle[i] > targetAngle[i])
        currentAngle[i] = max(currentAngle[i] - STEP_DEG, targetAngle[i]);
      else continue;
      servos[i].write(currentAngle[i]);
    }
  }
}