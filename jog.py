"""
Keyboard teleop for the ESP32 robot arm.

    pip install pyserial
    python arm_keyboard.py              (auto-detects the port)
    python arm_keyboard.py --port COM7  (or /dev/ttyUSB0 on Linux/Mac)

Keys (no Enter needed - just press; hold to keep moving):
    A / D   Base      left / right
    W / S   Shoulder  up / down
    R / F   Elbow     up / down
    T / G   Wrist     up / down
    Q / E   Gripper   open / close
    + / -   bigger / smaller step size
    H       home all joints        P   print positions
    X       quit
"""
import argparse
import sys
import time

import serial
from serial.tools import list_ports

# Keep these in sync with ANGLE_MIN / ANGLE_MAX / ANGLE_HOME in the .ino
# (pins live only in the .ino: B=P13, S=P26, E=P14, W=P27, C=P15)
JOINTS = {
    #       name        min  max  home
    "B": ["Base",      0,   180, 90],
    "S": ["Shoulder",  15,  165, 90],
    "E": ["Elbow",     0,   180, 90],
    "W": ["Wrist",     0,   180, 90],
    "C": ["Gripper",   20,  160, 90],
}

# key -> (joint, direction). Flip a sign if a joint moves the "wrong" way.
KEYMAP = {
    "a": ("B", +1), "d": ("B", -1),
    "w": ("S", +1), "s": ("S", -1),
    "r": ("E", +1), "f": ("E", -1),
    "t": ("W", +1), "g": ("W", -1),
    "q": ("C", +1), "e": ("C", -1),
}

STEP_SIZES = [1, 2, 5, 10, 15, 20]


# ---------- single-keypress reading (Windows + Linux/Mac) ----------
if sys.platform.startswith("win"):
    import msvcrt

    def get_key():
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):   # arrow/function keys: swallow 2nd byte
            msvcrt.getwch()
            return ""
        return ch.lower()

    class RawTerminal:
        def __enter__(self): return self
        def __exit__(self, *a): pass
else:
    import termios
    import tty

    def get_key():
        return sys.stdin.read(1).lower()

    class RawTerminal:
        def __enter__(self):
            self.fd = sys.stdin.fileno()
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
            return self

        def __exit__(self, *a):
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


# ---------- serial helpers ----------
def find_port():
    ports = list(list_ports.comports())
    hints = ("CP210", "CH340", "CH910", "USB", "UART", "SLAB", "wchusb")
    for p in ports:
        text = f"{p.device} {p.description} {p.hwid}"
        if any(h.lower() in text.lower() for h in hints):
            return p.device
    if ports:
        return ports[0].device
    sys.exit("No serial port found. Plug in the ESP32 or pass --port.")


def drain(ser):
    """Print anything the ESP32 has sent back."""
    while ser.in_waiting:
        line = ser.readline().decode(errors="ignore").strip()
        if line.startswith(("ERR", "READY", "POS")):
            print(f"\r  [esp32] {line}" + " " * 20)


def send(ser, cmd):
    ser.write((cmd + "\n").encode())


def status_line(angles, step):
    parts = "  ".join(f"{JOINTS[j][0]}:{angles[j]:>3}" for j in JOINTS)
    print(f"\r{parts}  | step {step:>2}°   ", end="", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", help="e.g. COM7 or /dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    args = ap.parse_args()

    port = args.port or find_port()
    print(f"Connecting to {port} ...")
    ser = serial.Serial(port, args.baud, timeout=0.1)
    time.sleep(2.5)          # ESP32 resets when the port opens; let it boot
    drain(ser)

    angles = {j: JOINTS[j][3] for j in JOINTS}
    step_idx = STEP_SIZES.index(5)

    print(__doc__.split("Keys", 1)[1].split("\n", 1)[1])
    send(ser, "H")
    status_line(angles, STEP_SIZES[step_idx])

    try:
        with RawTerminal():
            while True:
                key = get_key()
                if not key:
                    continue

                if key in KEYMAP:
                    j, direction = KEYMAP[key]
                    _, lo, hi, _ = JOINTS[j]
                    new = max(lo, min(hi, angles[j] + direction * STEP_SIZES[step_idx]))
                    if new != angles[j]:
                        angles[j] = new
                        send(ser, f"{j}{new}")
                elif key in ("+", "="):
                    step_idx = min(step_idx + 1, len(STEP_SIZES) - 1)
                elif key in ("-", "_"):
                    step_idx = max(step_idx - 1, 0)
                elif key == "h":
                    angles = {j: JOINTS[j][3] for j in JOINTS}
                    send(ser, "H")
                elif key == "p":
                    send(ser, "?")
                    time.sleep(0.1)
                elif key in ("x", "\x1b", "\x03"):
                    break

                drain(ser)
                status_line(angles, STEP_SIZES[step_idx])
    except KeyboardInterrupt:
        pass
    finally:
        print("\nBye - arm left in its current position.")
        ser.close()


if __name__ == "__main__":
    main()