# 5-DOF-Robotic-Arm-Basic

ESP32-based 5-DOF robotic arm with camera-driven object location.

## Contents

- `base_test/` — Arduino sketch for basic servo/base motion testing
- `CameraWebServer/` — ESP32-CAM web server sketch (streams frames over WiFi)
- `jog.py` — manual joint jogging over serial
- `locate.py` — single-shot object location from a captured frame
- `owl_locate.py` — OWL-based object detection and arm targeting
- `vision_test.py` — vision pipeline smoke test

## Setup

Python scripts read Azure OpenAI credentials from an external env file
(`AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT_MAIN`).

```
python -m venv .venv
.venv/Scripts/activate
pip install openai pillow
```
