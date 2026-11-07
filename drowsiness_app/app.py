from flask import Flask, render_template, redirect, jsonify, request, Response, send_from_directory
from ultralytics import YOLO
import cv2
import csv
import glob
import hmac
import io
from functools import wraps
import math
import os
import pickle
import queue
from collections import deque
import itertools
import random
import re
import shutil
import subprocess
import numpy as np
import threading
import serial
import json
import time
import wave                                   # <-- NEW
from datetime import datetime
from PIL import Image as PILImage, ImageOps
from picamera2 import Picamera2
from gpiozero import OutputDevice             # <-- NEW

app = Flask(__name__)

# ── Config ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH       = os.path.join(BASE_DIR, "..", "runs", "detect", "drowsiness_640", "weights", "best.pt")
DRIVER_DB_FILE   = os.path.join(BASE_DIR, "drivers_database.pkl")
ATTENDANCE_CSV   = os.path.join(BASE_DIR, "attendance_log.csv")
DRIVER_INFO_CSV  = os.path.join(BASE_DIR, "drivers_info.csv")
LOCK_FILE        = os.path.join(BASE_DIR, "system_lock.json")

# Records for the admin page
BREATH_LOG_CSV   = os.path.join(BASE_DIR, "breath_log.csv")      # Breathalyzer results + lock/reset logs
DROWSY_LOG_CSV   = os.path.join(BASE_DIR, "drowsy_log.csv")      # Warnings and drowsy alerts
GPS_LOG_CSV      = os.path.join(BASE_DIR, "gps_log.csv")         # Vehicle tracking (VK-162)
GPS_LOG_INTERVAL_SEC = 10                                        # How often to save position while moving

# ── Drowsy Alert Recording (Live Cam) ──────────────────
RECORDINGS_DIR     = os.path.join(BASE_DIR, "recordings")        # recordings/<date>/<Name>_<time>.mp4
RECORDINGS_LOG_CSV = os.path.join(BASE_DIR, "recordings_log.csv")
RECORD_FPS         = 10        # Frames per second for the video
RECORD_MAX_SEC     = 120       # Maximum length of a single video
RECORD_AFTER_SEC   = 3         # Continue recording for this long after the driver wakes up
RECORD_MIN_FREE_MB = 500       # Stop recording if free disk space falls below this
RECORD_KEEP_MAX    = 300       # Maximum number of videos to keep (deletes the oldest ones)

DRIVER_PHOTOS    = os.path.join(BASE_DIR, "..", "registered_drivers")
NEW_DRIVERS_DIR  = os.path.join(BASE_DIR, "new_drivers")

DROWSY_CLASSES   = ['drowsy', 'eye_closed_L', 'eye_closed_R', 'eye_close_L',
                    'mouth_open', 'head_down', 'head_tilt_L', 'head_tilt_R']
AWAKE_CLASSES    = ['awake', 'eye_open_L', 'eye_open_R', 'mouth_closed', 'head_normal']
KNOWN_CLASSES    = set(DROWSY_CLASSES) | set(AWAKE_CLASSES)

# ── Admin (for unlocking engine lock and accessing admin page) ──
# PIN for /admin and /api/unlock_system. Change it! Ex: ALERTX_ADMIN_PIN=5678 python3 app.py
ADMIN_PIN       = os.environ.get("ALERTX_ADMIN_PIN", "1234")
UNLOCK_WAIT_SEC = 5      # How long to wait for ESP32 confirmation on reset
LOCK_REASSERT_SEC = 2.0  # While locked on disk but ESP32 is not LOCKOUT, keep re-sending "LOCK" command
# LOCK FLOW: alcohol detected -> LOCKED until an ADMIN resets it -> THEN a cooldown -> THEN retest.
LOCKOUT_COOLDOWN_SEC = 30   # Cooldown AFTER the admin reset. Must match RESET_COOLDOWN_MS (30000) on the ESP32

# ── SMS (SIM800C) Config ───────────────────────────────
# SIM number inside the SIM800C (this is the SENDER, not the recipient)
GSM_SIM_NUMBER = "09983469694"

# Numbers to receive alerts (RECIPIENT). Can be 09XXXXXXXXX or +639XXXXXXXXX
SMS_RECIPIENTS = [n.strip() for n in
                  os.environ.get("SMS_RECIPIENTS", "09913122038").split(",") if n.strip()]
SMS_POLL_SEC = 3                 # How often to check for incoming SMS (for "RESET")
SMS_ON_STARTUP = True            # Send "SYSTEM ONLINE" SMS when the system boots up
SMS_NOTIFY_SESSION_END = os.environ.get("SMS_NOTIFY_SESSION_END", "1") == "1"   # SMS when the session ends (time-out)
SMS_NOTIFY_ENGINE_OFF  = os.environ.get("SMS_NOTIFY_ENGINE_OFF", "1") == "1"    # SMS when the ignition is turned off while monitoring
SMS_ENGINE_OFF_COOLDOWN_SEC = 20
SMS_NOTIFY_TIMEIN = os.environ.get("SMS_NOTIFY_TIMEIN", "1") == "1"   # "DRIVER VERIFICATION LOG (TIME-IN)" when face matches
STARTUP_GPS_WAIT_SEC = 15        # Maximum wait for GPS fix before sending startup SMS
SMS_DROWSY_COOLDOWN_SEC = 60     # Minimum interval between drowsy SMS alerts (anti-spam)
SMS_MAX_CHARS = 612              # Up to 4 parts for long SMS
SMS_LONG_MODE = os.environ.get("SMS_LONG_MODE", "pdu")   # "pdu" = concatenated long SMS | "split" = separate texts
GSM_BAUDRATES = (9600, 115200)
GSM_PORT_OVERRIDE = os.environ.get("GSM_PORT")   # Optional, ex. "/dev/ttyUSB0"
GSM_SMSC = os.environ.get("GSM_SMSC")            # Optional: SMS center number of the SIM

# ── GPS (VK-162 USB GPS) Config ────────────────────────
GPS_PORT_OVERRIDE = os.environ.get("GPS_PORT")
GPS_BAUD          = int(os.environ.get("GPS_BAUD", "9600"))
GPS_BAUD_FALLBACKS = (4800, 38400, 115200)
GPS_STALE_SEC     = 5
LAST_POS_FILE     = os.path.join(BASE_DIR, "last_gps_position.json")   # Last known position (survives reboot)
# SMS location format: "0" = plain coordinates (not blocked by carriers) | "1" = Google Maps link (SMS_LOCATION_LINK=1 python3 app.py)
SMS_LOCATION_LINK = os.environ.get("SMS_LOCATION_LINK", "0") == "1"
GPS_SAVE_INTERVAL_SEC = 20       # How often to save the last known position to disk
SMS_LOCATION_COOLDOWN_SEC = 15

# ── Face Scan ──────────────────────────────────────────
SCAN_TIMEOUT_SEC     = 20      # How long to scan for a face per tap
FACE_MATCH_THRESHOLD = 0.60    # Lower = stricter match

# ── Grace Period & Time Format ─────────────────────────
GRACE_SECONDS = 60     # Must match RETEST_GRACE_MS (60000) on the ESP32
GRACE_BUFFER  = 2      # Buffer before logging attendance (ensures ESP32 grace is finished)
TIME_FMT = "%I:%M:%S %p"
DATE_FMT = "%Y-%m-%d"

# ── DROWSINESS (Adjust settings here) ──────────────────
DETECT_CONF        = 0.25   # Lower = more sensitive (even from far away). More false hits, so there is time + ratio confirmation below
WARN_SECONDS       = 5.0    # Continuous eye closure / head down / drowsy signs for 5s = WARNING
DROWSY_SECONDS     = 10.0   # Continuous 10s (confirmed) = DROWSY ALERT (buzzer + voice + SMS)
WARN_EYES_SEC      = WARN_SECONDS
WARN_HEAD_SEC      = WARN_SECONDS
WARN_YAWN_SEC      = WARN_SECONDS   # Yawning warning is also 5s (not immediate). The yawn count (stats) uses YAWN_COUNT_SEC
SLEEP_GAP_SEC      = 1.2    # Buffer for blinking / brief tracking loss
AWAKE_CLEAR_SEC    = 0.5    # How quickly the count resets when eyes are open
RECOVER_SEC        = 3.0    # If there is no clear "eyes open" detection, how many seconds of not-sleeping before the alert stops
NO_DRIVER_SEC      = 3.0    # No person detected for this long = "No driver"
FACELOST_MAX_SEC   = 20.0   # Terminate alert if face is lost during an alert for this long
YAWN_COUNT_SEC     = 1.5    # A yawn of >=1.5s is only counted in stats (no sound)
RECOVER_EYES_SEC   = 0.6    # Eyes open within 0.6s = awake immediately, sound/vibration stops
WARN_CONFIRM_RATIO = 0.60   # Within 5s, >=60% of frames must show the sign before a WARNING
DROWSY_CONFIRM_RATIO = 0.70 # Within 10s, >=70% of frames must show sleeping/closed eyes before a DROWSY ALERT
HIST_MIN_FRAMES    = 6      # Minimum frames in the window before trusting it (avoids false alarms when detection is sparse)
ZOOM_ENABLED       = True   # When no eyes are visible (far away), repeat detection on a zoomed crop
ZOOM_CENTER_FRAC   = 0.65   # If nothing was detected, zoom into the center (65% of the frame)
ZOOM_PAD           = 0.5    # Extra margin around the detected face before zooming
WARN_BEEP_REPEAT   = 1.5    # During WARNING: buzzer beeps every X seconds (buzzer only, no high-level alarm)

# ── Session State ──────────────────────────────────────
session = {
    "driver_name": None,
    "plate_number": None,
    "login_time": None,        # String for display
    "login_dt": None,          # Actual face scan datetime object
    "login_lat": "",           # GPS latitude at login
    "login_lon": "",           # GPS longitude at login
    "alerts": [],              # DROWSY ALERTS (timestamps)
    "warn_events": [],         # WARNINGS (eyes closed, yawn, head down)
    "yawns": 0,
    "videos": 0,               # Number of videos recorded in this session
    "alcohol_result": None,    # "CLEAR" / "DETECTED" / None
    "alcohol_level": "",
    "current_status": "AWAKE",
    "started": False,          # Passed breathalyzer / engine started?
    "engine_off_at": None,     # Time engine turned off (start of grace period)
    "logged": False,           # Already logged to attendance?
    "summary": None,
}
session_lock = threading.RLock()
csv_lock = threading.Lock()

last_drowsy_sms = 0.0
last_engine_off_sms = 0.0
last_location_sms = 0.0

# ── Load YOLO model ────────────────────────────────────
model = YOLO('best.pt')

# ── Global Camera: SINGLE capture thread ────────────────
class CameraHub:
    STALE_SEC    = 3.0     # Restart camera if no new frame for this long
    JPEG_QUALITY = 80

    def __init__(self):
        self.cam = None
        self.frame = None          # BGR format, immutable after set
        self.seq = 0
        self.ts = 0.0
        self.jpeg = None           # Mirrored JPEG for streaming
        self.jpeg_seq = -1
        self.fails = 0
        self.cond = threading.Condition()
        self.jpeg_lock = threading.Lock()
        self._open()
        threading.Thread(target=self._run, daemon=True).start()

    def _open(self):
        try:
            if self.cam is not None:
                try:
                    self.cam.stop()
                except Exception:
                    pass
                try:
                    self.cam.close()
                except Exception:
                    pass
                self.cam = None
                time.sleep(1.0)
            cam = Picamera2()
            cam.configure(cam.create_preview_configuration(main={"size": (640, 480)}))
            cam.start()
            self.cam = cam
            self.fails = 0
            print("[camera] started")
            return True
        except Exception as e:
            print(f"[camera] cannot start: {e}")
            self.cam = None
            return False

    def _run(self):
        while True:
            try:
                if self.cam is None:
                    if not self._open():
                        time.sleep(2)
                        continue
                raw = self.cam.capture_array()
                if raw is None:
                    raise RuntimeError("empty frame")
                bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
                with self.cond:
                    self.frame = bgr
                    self.ts = time.time()
                    self.seq += 1
                    self.fails = 0
                    self.cond.notify_all()
            except Exception as e:
                self.fails += 1
                print(f"[camera] capture error ({self.fails}): {e}")
                time.sleep(0.2)
                if self.fails >= 5 or (self.ts and time.time() - self.ts > self.STALE_SEC):
                    print("[camera] restarting camera...")
                    self._open()

    def read(self, max_age=2.0):
        """Latest BGR frame, or None if expired."""
        f, ts = self.frame, self.ts
        if f is None or time.time() - ts > max_age:
            return None
        return f

    def wait_new(self, last_seq, timeout=2.0):
        with self.cond:
            if self.seq == last_seq:
                self.cond.wait(timeout)
            return self.seq

    def get_jpeg(self):
        """(seq, bytes). Encodes frame to JPEG only once per new frame."""
        with self.jpeg_lock:
            f, seq = self.frame, self.seq
            if f is None:
                return seq, None
            if seq != self.jpeg_seq:
                ok, buf = cv2.imencode('.jpg', cv2.flip(f, 1), [cv2.IMWRITE_JPEG_QUALITY, self.JPEG_QUALITY])
                if ok:
                    self.jpeg, self.jpeg_seq = buf.tobytes(), seq
            return self.jpeg_seq, self.jpeg

camera = CameraHub()

# ── Bluetooth Serial Interlock Setup ────────────────────
esp32_data = {"state": "IDLE", "val": 0}
esp32_last_seen = 0
BT_STALE_AFTER_SEC = 5

breath_session = {
    "awaiting": False,
    "result": None,   # "passed" / "locked" / None
    "peak": 0,
    "max": 0,         # Highest reading in current test
}

lock_ctl = {"unlocking_until": 0.0, "last_cmd": 0.0, "source": "", "source_at": 0.0}

ser = None
ser_lock = threading.Lock()

def connect_bluetooth():
    global ser
    try:
        ser = serial.Serial('/dev/rfcomm0', 115200, timeout=1)
        print("Bluetooth Serial connected to /dev/rfcomm0")
    except Exception as e:
        print(f"Bluetooth Serial connection waiting: {e}")

# ── Persistent Lock Functions ───────────────────────────
def set_system_lock(locked: bool, reason="", **extra):
    """Saves lock state permanently using atomic write + fsync."""
    data = {"locked": locked, "reason": reason,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": time.time(), **extra}
    tmp = LOCK_FILE + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, LOCK_FILE)
    except Exception as e:
        print(f"Error saving lock file: {e}")

_lock_cache = {"sig": "unset", "info": None}

def get_lock_info():
    """Reads lock details from disk with a performance cache."""
    try:
        st = os.stat(LOCK_FILE)
        sig = (st.st_mtime_ns, st.st_size)
    except FileNotFoundError:
        sig = None
    except Exception:
        sig = "err"
    if sig == _lock_cache["sig"] and _lock_cache["info"] is not None:
        return dict(_lock_cache["info"])

    info = {"locked": False, "reason": "", "time": None, "driver": None, "plate": None, "level": 0}
    if sig is not None:
        try:
            with open(LOCK_FILE, "r") as f:
                info.update(json.load(f))
        except Exception:
            info.update(locked=True, reason="Lock file unreadable")
    _lock_cache.update(sig=sig, info=dict(info))
    return info

def get_persisted_lock():
    return bool(get_lock_info().get("locked"))

def clear_persisted_lock(source=""):
    was_locked = os.path.exists(LOCK_FILE)
    if was_locked:
        try:
            os.remove(LOCK_FILE)
        except Exception as e:
            print(f"Cannot remove lock file: {e}")
    breath_session.update(awaiting=False, result=None, peak=0, max=0)
    session["engine_off_at"] = None
    if was_locked:
        log_breath("LOCK RESET", note=f"Reset by {source or 'admin'}")
    print(f"[lock] unlocked ({source})")

def is_system_locked():
    esp_locked = str(esp32_data.get("state", "")).upper() == "LOCKOUT"
    return esp_locked or get_persisted_lock()

def is_engine_active():
    return str(esp32_data.get("state", "")).upper() == "PASSED" and not get_persisted_lock()

def monitoring_on():
    return is_engine_active() and not session["logged"]

def send_to_esp32(command):
    global ser
    with ser_lock:
        if ser and ser.is_open:
            try:
                ser.write((command + "\n").encode())
                return True
            except Exception as e:
                print(f"ESP32 write error ({command}): {e}")
    return False

def admin_ok():
    pin = request.headers.get("X-Admin-Pin") or request.values.get("pin") or ""
    return hmac.compare_digest(pin.encode("utf-8"), ADMIN_PIN.encode("utf-8"))

def admin_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not admin_ok():
            return jsonify({"status": "unauthorized", "message": "Admin PIN required"}), 401
        return fn(*args, **kwargs)
    return wrapper

# ── CSV Log Helpers ────────────────────────────────────
BREATH_HEADER = ["Date", "Time", "Driver", "Plate", "Result", "Level", "Latitude", "Longitude", "Note"]
DROWSY_HEADER = ["Date", "Time", "Driver", "Plate", "Type", "Detail", "Latitude", "Longitude"]
GPS_HEADER    = ["Date", "Time", "Driver", "Plate", "Latitude", "Longitude", "Speed", "Satellites"]
TRIPS_HEADER  = ["Name", "Plate", "Login", "Logout", "Duration", "Alerts", "Rating", "Date",
                 "Warnings", "Yawns", "Alcohol Result", "Alcohol Level", "Login Lat", "Login Lon", "Status"]
VIDEO_HEADER  = ["Date", "Time", "Driver", "Plate", "File", "Duration", "Latitude", "Longitude", "Detail"]
ADMIN_LOGS = {
    "breath": (BREATH_LOG_CSV,     BREATH_HEADER),
    "drowsy": (DROWSY_LOG_CSV,     DROWSY_HEADER),
    "gps":    (GPS_LOG_CSV,        GPS_HEADER),
    "trips":  (ATTENDANCE_CSV,     TRIPS_HEADER),
    "videos": (RECORDINGS_LOG_CSV, VIDEO_HEADER),
}

def append_csv(path, header, row):
    try:
        with csv_lock:
            exists = os.path.isfile(path)
            with open(path, "a", newline="") as f:
                w = csv.writer(f)
                if not exists:
                    w.writerow(header)
                w.writerow(row)
    except Exception as e:
        print(f"[log] cannot write {path}: {e}")

def read_csv_rows(path):
    if not os.path.isfile(path):
        return []
    try:
        with csv_lock:
            with open(path, newline="") as f:
                return list(csv.DictReader(f))
    except Exception as e:
        print(f"[log] cannot read {path}: {e}")
        return []

def migrate_csv(path, header):
    try:
        if not os.path.isfile(path):
            return
        with csv_lock:
            with open(path, newline="") as f:
                rows = list(csv.reader(f))
            if not rows or rows[0] == header:
                return
            old = rows[0]
            fixed = []
            for r in rows[1:]:
                d = dict(zip(old, r))
                fixed.append([d.get(c, "") for c in header])
            tmp = path + ".tmp"
            with open(tmp, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(fixed)
            os.replace(tmp, path)
        print(f"[log] updated columns of {os.path.basename(path)}")
    except Exception as e:
        print(f"[log] migrate error {path}: {e}")

migrate_csv(ATTENDANCE_CSV, TRIPS_HEADER)
os.makedirs(RECORDINGS_DIR, exist_ok=True)

def gps_pos():
    try:
        s = gps.snapshot()
        if s["has_position"]:
            return round(s["lat"], 6), round(s["lon"], 6)
    except Exception:
        pass
    return "", ""

def maps_link(lat, lon):
    if lat in ("", None) or lon in ("", None):
        return ""
    return f"https://maps.google.com/?q={lat},{lon}"

def _who():
    return (session["driver_name"] or "Unknown"), (session["plate_number"] or "N/A")

def log_breath(result, level="", note=""):
    now = datetime.now()
    name, plate = _who()
    lat, lon = gps_pos()
    append_csv(BREATH_LOG_CSV, BREATH_HEADER,
               [now.strftime(DATE_FMT), now.strftime(TIME_FMT), name, plate, result, level, lat, lon, note])

def log_drowsy(kind, detail=""):
    now = datetime.now()
    name, plate = _who()
    lat, lon = gps_pos()
    append_csv(DROWSY_LOG_CSV, DROWSY_HEADER,
               [now.strftime(DATE_FMT), now.strftime(TIME_FMT), name, plate, kind, detail, lat, lon])

def enforce_lock():
    now = time.time()
    if now < lock_ctl["unlocking_until"] or now - lock_ctl["last_cmd"] < LOCK_REASSERT_SEC:
        return
    lock_ctl["last_cmd"] = now
    if get_persisted_lock() and str(esp32_data.get("state", "")).upper() != "LOCKOUT":
        send_to_esp32("LOCK")

# ── Drowsy Alert Incident Recorder ──────────────────────
def prune_recordings():
    try:
        files = []
        for root, _, names in os.walk(RECORDINGS_DIR):
            for n in names:
                if n.lower().endswith(".mp4"):
                    p = os.path.join(root, n)
                    files.append((os.path.getmtime(p), p))
        files.sort()
        for _, p in files[:max(0, len(files) - RECORD_KEEP_MAX)]:
            os.remove(p)
            print(f"[rec] deleted old video {p}")
    except Exception as e:
        print(f"[rec] prune error: {e}")

class IncidentRecorder:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = False
        self.stop_at = None

    def start(self, driver, plate, detail=""):
        with self.lock:
            if self.active:
                self.stop_at = None
                return True
            try:
                free_mb = shutil.disk_usage(BASE_DIR).free / 1048576
            except Exception:
                free_mb = 99999
            if free_mb < RECORD_MIN_FREE_MB:
                print(f"[rec] not recording: only {int(free_mb)} MB free")
                return False
            self.active = True
            self.stop_at = None
        lat, lon = gps_pos()
        threading.Thread(target=self._run, args=(driver, plate, detail, lat, lon), daemon=True).start()
        return True

    def stop(self):
        with self.lock:
            if self.active and self.stop_at is None:
                self.stop_at = time.time() + RECORD_AFTER_SEC

    def _run(self, driver, plate, detail, lat, lon):
        t0 = datetime.now()
        day = t0.strftime(DATE_FMT)
        day_dir = os.path.join(RECORDINGS_DIR, day)
        os.makedirs(day_dir, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", driver or "Unknown").strip("_") or "Unknown"
        base = f"{safe}_{t0:%H%M%S}"
        raw_path = os.path.join(day_dir, base + "_raw.mp4")
        final_path = os.path.join(day_dir, base + ".mp4")
        label_who = (driver or "Unknown").encode("ascii", "ignore").decode() or "Driver"
        label_plate = (plate or "N/A").encode("ascii", "ignore").decode()
        writer, frames = None, 0
        start = time.time()
        try:
            interval = 1.0 / RECORD_FPS
            next_t, last_seq = 0.0, -1
            while True:
                now = time.time()
                if self.stop_at is not None and now >= self.stop_at:
                    break
                if now - start >= RECORD_MAX_SEC:
                    break
                seq = camera.wait_new(last_seq, timeout=1.0)
                if seq == last_seq:
                    continue
                last_seq = seq
                if now < next_t:
                    continue
                frame = camera.read()
                if frame is None:
                    continue
                next_t = now + interval
                img = frame.copy()
                stamp = datetime.now().strftime("%Y-%m-%d  %I:%M:%S %p")
                cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
                cv2.putText(img, f"{stamp} | {label_who} | {label_plate} | DROWSY ALERT",
                            (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                if writer is None:
                    h, w = img.shape[:2]
                    writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), RECORD_FPS, (w, h))
                    if not writer.isOpened():
                        raise RuntimeError("VideoWriter cannot open")
                writer.write(img)
                frames += 1
        except Exception as e:
            print(f"[rec] error: {e}")
        finally:
            if writer is not None:
                writer.release()
            with self.lock:
                self.active = False
                self.stop_at = None

        dur = max(time.time() - start, 0.1)
        if frames < 5 or not os.path.isfile(raw_path):
            try:
                os.remove(raw_path)
            except Exception:
                pass
            return

        converted = False
        ff = shutil.which("ffmpeg")
        if ff:
            real_fps = frames / dur
            factor = RECORD_FPS / real_fps
            vf = "format=yuv420p"
            if abs(factor - 1) > 0.1:
                vf = f"setpts={factor:.3f}*PTS,format=yuv420p"
            try:
                subprocess.run([ff, "-y", "-loglevel", "error", "-i", raw_path, "-vf", vf, "-r", str(RECORD_FPS),
                                "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
                                "-movflags", "+faststart", final_path],
                               timeout=240, check=True)
                os.remove(raw_path)
                converted = True
            except Exception as e:
                print(f"[rec] ffmpeg failed: {e}")
        if not converted:
            os.replace(raw_path, final_path)

        rel = f"{day}/{base}.mp4"
        append_csv(RECORDINGS_LOG_CSV, VIDEO_HEADER,
                   [day, t0.strftime(TIME_FMT), driver or "Unknown", plate or "N/A",
                    rel, f"{int(dur)}s", lat, lon, detail])
        with session_lock:
            session["videos"] += 1
        print(f"[rec] saved {rel} ({int(dur)}s)")
        prune_recordings()

recorder = IncidentRecorder()

# ── GPS VK-162 (USB, NMEA) ─────────────────────────────
def _num(x, cast=float, default=None):
    try:
        return cast(x)
    except (TypeError, ValueError):
        return default

gsm_boot_done = threading.Event()

def _usb_ids():
    out = {}
    for p in glob.glob("/dev/serial/by-id/*"):
        try:
            out[os.path.realpath(p)] = os.path.basename(p).lower()
        except Exception:
            pass
    return out

def _looks_like_gps(port):
    name = _usb_ids().get(port, "")
    return any(k in name for k in ("gps", "u-blox", "ublox", "gnss"))

class GPSReader:
    def __init__(self):
        self.port = None
        self.baud = None
        self.dev = None
        self.last_error = "GPS not started"
        self.scan_log = []
        self.lock = threading.Lock()
        self.first_scan_done = threading.Event()
        self.data = {"fix": False, "lat": None, "lon": None, "alt": None,
                     "speed_kmh": None, "course": None, "sats": 0, "hdop": None,
                     "updated": 0.0, "last_fix": 0.0}
        self._last_save = 0.0
        self.saved = self._load_saved()

    @staticmethod
    def _load_saved():
        try:
            with open(LAST_POS_FILE, "r") as f:
                d = json.load(f)
            if d.get("lat") is not None and d.get("lon") is not None:
                return d
        except Exception:
            pass
        return None

    def _remember(self, lat, lon, now):
        """Save the last good position to disk (throttled) so SMS never says 'unavailable' after a fix was seen once."""
        if now - self._last_save < GPS_SAVE_INTERVAL_SEC:
            return
        self._last_save = now
        d = {"lat": round(lat, 6), "lon": round(lon, 6),
             "time_str": datetime.now().strftime("%b %d %I:%M %p")}
        self.saved = d
        try:
            tmp = LAST_POS_FILE + ".tmp"
            with open(tmp, "w") as f:
                json.dump(d, f)
            os.replace(tmp, LAST_POS_FILE)
        except Exception as e:
            print(f"[GPS] cannot save last position: {e}")

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    @staticmethod
    def _checksum_ok(line):
        if not line.startswith("$") or "*" not in line:
            return False
        body, _, cs = line[1:].partition("*")
        calc = 0
        for ch in body:
            calc ^= ord(ch)
        try:
            return calc == int(cs[:2], 16)
        except ValueError:
            return False

    @staticmethod
    def _to_deg(value, hemi):
        if not value or "." not in value:
            return None
        dot = value.index(".")
        deg = _num(value[:dot - 2])
        minutes = _num(value[dot - 2:])
        if deg is None or minutes is None:
            return None
        d = deg + minutes / 60.0
        return -d if hemi in ("S", "W") else d

    def _parse(self, line):
        if not self._checksum_ok(line):
            return
        f = line[1:line.index("*")].split(",")
        kind = f[0][-3:]
        now = time.time()
        with self.lock:
            if kind == "GGA" and len(f) >= 10:
                quality = _num(f[6], int, 0)
                self.data["sats"] = _num(f[7], int, 0)
                self.data["hdop"] = _num(f[8])
                if quality and quality > 0:
                    lat = self._to_deg(f[2], f[3])
                    lon = self._to_deg(f[4], f[5])
                    if lat is not None and lon is not None:
                        self.data.update(lat=lat, lon=lon, alt=_num(f[9]), fix=True, last_fix=now)
                        self._remember(lat, lon, now)
                else:
                    self.data["fix"] = False
                self.data["updated"] = now

            elif kind == "RMC" and len(f) >= 9:
                if f[2] == "A":
                    lat = self._to_deg(f[3], f[4])
                    lon = self._to_deg(f[5], f[6])
                    if lat is not None and lon is not None:
                        knots = _num(f[7], float, 0.0)
                        self.data.update(lat=lat, lon=lon, fix=True, last_fix=now,
                                         speed_kmh=round(knots * 1.852, 1),
                                         course=_num(f[8]))
                        self._remember(lat, lon, now)
                else:
                    self.data["fix"] = False
                self.data["updated"] = now

    def _candidate_ports(self):
        if GPS_PORT_OVERRIDE:
            return [GPS_PORT_OVERRIDE]
        ports = []
        for p in sorted(glob.glob("/dev/serial/by-id/*")):
            if any(k in p.lower() for k in ("gps", "u-blox", "ublox")):
                ports.append(os.path.realpath(p))
        ports += sorted(glob.glob("/dev/ttyACM*")) + sorted(glob.glob("/dev/ttyUSB*"))
        g = globals().get("gsm")
        seen, out = set(), []
        for p in ports:
            if p in seen or (g is not None and p == g.port):
                continue
            seen.add(p)
            out.append(p)
        return out

    def _scan(self):
        log = []
        bauds = []
        for b in (GPS_BAUD,) + GPS_BAUD_FALLBACKS:
            if b not in bauds:
                bauds.append(b)
        ports = self._candidate_ports()
        if not ports:
            log.append("No /dev/ttyACM* or /dev/ttyUSB* found - is the VK-162 plugged in?")
        for port in ports:
            for baud in bauds:
                try:
                    dev = serial.Serial(port, baud, timeout=1)
                except Exception as e:
                    hint = " -> run: sudo usermod -aG dialout $USER (then reboot)" if "ermission" in str(e) else ""
                    log.append(f"{port}: cannot open ({e}){hint}")
                    break
                ok, got_bytes = False, False
                end = time.time() + (3 if baud == bauds[0] else 2)
                while time.time() < end:
                    try:
                        raw = dev.readline()
                    except Exception:
                        break
                    if raw:
                        got_bytes = True
                    line = raw.decode("ascii", errors="ignore").strip()
                    if line.startswith("$G") and self._checksum_ok(line):
                        ok = True
                        break
                if ok:
                    self.dev, self.port, self.baud, self.last_error = dev, port, baud, ""
                    log.append(f"{port}: NMEA OK @ {baud}")
                    self.scan_log = log
                    print(f"[GPS] VK-162 found on {port} @ {baud}")
                    return True
                log.append(f"{port} @ {baud}: " + ("data but not NMEA" if got_bytes else "no data"))
                try:
                    dev.close()
                except Exception:
                    pass
                if not got_bytes:
                    break
        self.scan_log = log
        self.last_error = "GPS not found (check USB / GPS_PORT / GPS_BAUD / ModemManager / gpsd)"
        return False

    def _close(self):
        try:
            if self.dev:
                self.dev.close()
        except Exception:
            pass
        self.dev = None
        self.port = None
        self.baud = None

    def _run(self):
        gsm_boot_done.wait(timeout=30)
        while True:
            if not self.dev:
                self._scan()
                self.first_scan_done.set()
                if not self.dev:
                    time.sleep(10)
                    continue
            try:
                raw = self.dev.readline()
                if raw:
                    line = raw.decode("ascii", errors="ignore").strip()
                    if line.startswith("$"):
                        self._parse(line)
            except Exception as e:
                self.last_error = f"read error: {e}"
                print(f"[GPS] {self.last_error}")
                self._close()
                time.sleep(2)

    def snapshot(self):
        with self.lock:
            d = dict(self.data)
        now = time.time()
        d["fix"] = bool(d["fix"]) and (now - d["updated"]) < GPS_STALE_SEC
        d["has_position"] = d["lat"] is not None
        d["age_sec"] = int(now - d["last_fix"]) if d["last_fix"] else None
        d["port"] = self.port
        d["baud"] = self.baud
        d["scan_log"] = list(self.scan_log)
        d["status"] = "no_device" if not self.port else ("fix" if d["fix"] else "no_fix")
        d["last_error"] = self.last_error
        d["maps_url"] = self.maps_url(d)
        return d

    @staticmethod
    def maps_url(d):
        if d.get("lat") is None or d.get("lon") is None:
            return None
        return f"https://maps.google.com/?q={d['lat']:.5f},{d['lon']:.5f}"

    @staticmethod
    def _loc_ref(lat, lon):
        if SMS_LOCATION_LINK:
            return f"https://maps.google.com/?q={lat:.5f},{lon:.5f}"
        return f"{lat:.5f}, {lon:.5f}"

    def location_text(self):
        snap = self.snapshot()
        if snap["has_position"]:
            ref = self._loc_ref(snap["lat"], snap["lon"])
            if snap["fix"]:
                return ref
            mins = max(1, (snap["age_sec"] or 0) // 60)
            return f"Last known ({mins} min ago) {ref}"
        saved = self.saved
        if saved:
            ref = self._loc_ref(saved["lat"], saved["lon"])
            when = saved.get("time_str", "")
            return f"Last known ({when}) {ref}" if when else f"Last known {ref}"
        if not snap["port"]:
            return "GPS not detected (check VK-162 USB)"
        return f"No satellite fix yet ({snap['sats']} sats) - GPS needs open sky"

gps = GPSReader()
gps.start()

# ── GSM SIM800C (USB Plug & Play) ──────────────────────
def normalize_ph_number(num):
    n = "".join(ch for ch in str(num) if ch.isdigit() or ch == "+")
    if n.startswith("+"):
        return n
    if n.startswith("09") and len(n) == 11:
        return "+63" + n[1:]
    if n.startswith("9") and len(n) == 10:
        return "+63" + n
    if n.startswith("63"):
        return "+" + n
    return n

def sanitize_sms(text, limit=SMS_MAX_CHARS):
    cleaned = text.encode("ascii", errors="ignore").decode("ascii")
    cleaned = cleaned.replace("\r", "").replace('"', "'")
    return cleaned[:limit]

_GSM7_EXT = {"^": 0x14, "{": 0x28, "}": 0x29, "\\": 0x2F, "[": 0x3C, "~": 0x3D, "]": 0x3E, "|": 0x40}

def _gsm7_char(ch):
    o = ord(ch)
    if ch == "@": return [0x00]
    if ch == "$": return [0x02]
    if ch == "_": return [0x11]
    if ch in "\n\r": return [o]
    if ch in _GSM7_EXT: return [0x1B, _GSM7_EXT[ch]]
    if 0x20 <= o <= 0x7E and ch != "`": return [o]
    return [ord("?")]

def gsm7_split(text, single=160, multi=153):
    chars = [_gsm7_char(c) for c in text]
    if sum(len(c) for c in chars) <= single:
        return [[x for c in chars for x in c]]
    parts, cur = [], []
    for c in chars:
        if len(cur) + len(c) > multi:
            parts.append(cur)
            cur = []
        cur += c
    if cur:
        parts.append(cur)
    return parts

def _pack_septets(septets, prefix=b"", fill_bits=0):
    bits = []
    for byte in prefix:
        bits += [(byte >> i) & 1 for i in range(8)]
    bits += [0] * fill_bits
    for sp in septets:
        bits += [(sp >> i) & 1 for i in range(7)]
    while len(bits) % 8:
        bits.append(0)
    out = bytearray()
    for i in range(0, len(bits), 8):
        out.append(sum(bits[i + j] << j for j in range(8)))
    return bytes(out)

def build_pdu(number, septets, udh=None):
    digits = "".join(ch for ch in str(number) if ch.isdigit())
    padded = digits + ("F" if len(digits) % 2 else "")
    addr = "".join(padded[i + 1] + padded[i] for i in range(0, len(padded), 2))
    pdu = bytearray([0x41 if udh else 0x01, 0x00, len(digits), 0x91]) + bytes.fromhex(addr) + bytes([0x00, 0x00])
    if udh:
        pdu += bytes([7 + len(septets)]) + _pack_septets(septets, prefix=udh, fill_bits=1)
    else:
        pdu += bytes([len(septets)]) + _pack_septets(septets)
    return ("00" + pdu.hex()).upper(), len(pdu)

def split_text_lines(text, limit=148):
    chunks, cur = [], ""
    for line in text.split("\n"):
        if cur and len(cur) + 1 + len(line) > limit:
            chunks.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        chunks.append(cur)
    n = len(chunks)
    return [f"({i}/{n})\n{c}" for i, c in enumerate(chunks, 1)] if n > 1 else chunks

def build_sms_plan(number, text):
    parts = gsm7_split(text)
    if len(parts) == 1:
        return [("text", number, text)]
    if SMS_LONG_MODE == "split":
        return [("text", number, chunk) for chunk in split_text_lines(text)]
    ref = random.randint(1, 255)
    plan = []
    for i, septets in enumerate(parts, 1):
        hexpdu, tlen = build_pdu(number, septets, bytes([0x05, 0x00, 0x03, ref, len(parts), i]))
        plan.append(("pdu", hexpdu, tlen))
    return plan

class AmbiguousSend(Exception):
    """SMS written to modem but not confirmed - do not retry to avoid duplicates."""

def parse_cmgl(resp):
    msgs, cur = [], None
    for line in resp.replace("\r", "").split("\n"):
        line = line.strip()
        if line.startswith("+CMGL:"):
            m = re.match(r'\+CMGL:\s*(\d+),"[^"]*","([^"]*)"', line)
            cur = {"index": int(m.group(1)), "sender": m.group(2), "text": ""} if m else None
            if cur:
                msgs.append(cur)
        elif cur and line and line not in ("OK", "ERROR") and not line.startswith("AT+"):
            cur["text"] += (" " if cur["text"] else "") + line
    return msgs

class GSMModem:
    DEDUPE_SEC = 20

    def __init__(self):
        self.port = None
        self.dev = None
        self.status = "disconnected"
        self.last_error = ""
        self.last_sent = None
        self.last_reg_ok = 0.0
        self.recent = {}
        self.io_lock = threading.Lock()
        self.q = queue.PriorityQueue()     # (priority, seq, queued_at, number, text): 0 = urgent, 1 = normal, 2 = low
        self._seq = itertools.count()
        self.busy = False
        threading.Thread(target=self._worker, daemon=True).start()

    def _read_until(self, markers, timeout):
        buf = b""
        end = time.time() + timeout
        while time.time() < end:
            chunk = self.dev.read(self.dev.in_waiting or 1)
            if chunk:
                buf += chunk
                if any(m in buf for m in markers):
                    break
        return buf.decode("utf-8", errors="ignore")

    def _cmd(self, cmd, expect=(b"OK", b"ERROR"), timeout=3):
        self.dev.reset_input_buffer()
        self.dev.write((cmd + "\r").encode())
        resp = self._read_until(expect, timeout)
        return ("OK" in resp and "ERROR" not in resp), resp

    def _candidate_ports(self):
        if GSM_PORT_OVERRIDE:
            return [GSM_PORT_OVERRIDE]
        ports = sorted(glob.glob("/dev/ttyUSB*")) + sorted(glob.glob("/dev/ttyACM*"))
        return [p for p in ports if p != gps.port and not _looks_like_gps(p)]

    def _probe(self, port):
        for baud in GSM_BAUDRATES:
            try:
                dev = serial.Serial(port, baud, timeout=0.3, write_timeout=2)
            except Exception:
                return None
            self.dev = dev
            try:
                time.sleep(0.2)
                for _ in range(3):
                    ok, _ = self._cmd("AT", timeout=1)
                    if ok:
                        return dev
            except Exception:
                pass
            try:
                dev.close()
            except Exception:
                pass
            self.dev = None
        return None

    def _init_modem(self):
        self._cmd("ATE0")
        self._cmd("AT+CMEE=1")
        ok, resp = self._cmd("AT+CPIN?", timeout=3)
        if "READY" not in resp:
            self.status = "no_sim"
            self.last_error = "SIM not ready (check SIM card / PIN)"
            return False
        self._cmd("AT+CMGF=1")
        self._cmd('AT+CSCS="GSM"')
        self._cmd('AT+CPMS="SM","SM","SM"')
        self._cmd("AT+CMGD=1,4", timeout=8)
        self._cmd("AT+CSMP=17,167,0,0")
        if GSM_SMSC:
            self._cmd(f'AT+CSCA="{GSM_SMSC}"')
        self.status = "searching_network"
        return True

    def _network_ready(self, wait=20):
        last = ""
        if self.status == "ready" and time.time() - self.last_reg_ok < 30:
            return True
        end = time.time() + wait
        while time.time() < end:
            ok, resp = self._cmd("AT+CREG?", timeout=2)
            last = resp
            if "+CREG:" in resp and (",1" in resp or ",5" in resp):
                self.status = "ready"
                self.last_reg_ok = time.time()
                return True
            time.sleep(1)
        _, csq = self._cmd("AT+CSQ", timeout=2)
        self.last_error = f"No network signal / not registered (CREG: {last.strip()!r}, CSQ: {csq.strip()!r})"
        self.status = "searching_network"
        return False

    def connect(self):
        self.close()
        for port in self._candidate_ports():
            if self._probe(port):
                self.port = port
                print(f"[GSM] Modem found on {port}")
                return self._init_modem()
        self.status = "disconnected"
        self.last_error = "GSM modem not found on USB"
        return False

    def close(self):
        try:
            if self.dev:
                self.dev.close()
        except Exception:
            pass
        self.dev = None
        self.port = None

    def _send_now(self, number, text):
        if not self.dev or not self.dev.is_open:
            raise RuntimeError("modem not connected")
        if not self._network_ready(wait=15):
            raise RuntimeError(self.last_error)
        self._cmd("AT+CMGF=1")
        self.dev.reset_input_buffer()
        self.dev.write(f'AT+CMGS="{number}"\r'.encode())
        resp = self._read_until((b">", b"ERROR"), 5)
        if ">" not in resp:
            self.dev.write(b"\x1b")
            raise RuntimeError(f"no '>' prompt: {resp.strip()}")
        self.dev.write(text.encode("ascii") + b"\x1a")
        resp = self._read_until((b"+CMGS:", b"ERROR"), 60)
        if "+CMGS:" not in resp:
            if "ERROR" in resp:
                raise RuntimeError(f"send failed: {resp.strip()}")
            raise AmbiguousSend(f"no +CMGS confirmation: {resp.strip()}")

    def _send_pdu(self, hexpdu, tpdu_len):
        if not self.dev or not self.dev.is_open:
            raise RuntimeError("modem not connected")
        if not self._network_ready(wait=15):
            raise RuntimeError(self.last_error)
        try:
            self._cmd("AT+CMGF=0")
            self.dev.reset_input_buffer()
            self.dev.write(f"AT+CMGS={tpdu_len}\r".encode())
            resp = self._read_until((b">", b"ERROR"), 5)
            if ">" not in resp:
                self.dev.write(b"\x1b")
                raise RuntimeError(f"no '>' prompt (PDU): {resp.strip()}")
            self.dev.write(hexpdu.encode("ascii") + b"\x1a")
            resp = self._read_until((b"+CMGS:", b"ERROR"), 60)
            if "+CMGS:" not in resp:
                if "ERROR" in resp:
                    raise RuntimeError(f"PDU send failed: {resp.strip()}")
                raise AmbiguousSend(f"no +CMGS confirmation (PDU): {resp.strip()}")
            self._read_until((b"OK", b"ERROR"), 3)
        finally:
            try:
                self._cmd("AT+CMGF=1")
            except Exception:
                pass

    def _send_item(self, item):
        if item[0] == "pdu":
            self._send_pdu(item[1], item[2])
        else:
            self._send_now(item[1], item[2])

    def _worker(self):
        while True:
            prio, _seq, t_enq, number, text = self.q.get()
            self.busy = True
            t_start = time.time()
            plan = build_sms_plan(number, text)
            done = 0
            sent = False
            for attempt in range(3):
                ambiguous = False
                with self.io_lock:
                    try:
                        if not self.dev or not self.dev.is_open:
                            if not self.connect():
                                raise RuntimeError(self.last_error)
                        while done < len(plan):
                            self._send_item(plan[done])
                            done += 1
                            if done < len(plan):
                                time.sleep(0.3)
                        sent = True
                    except AmbiguousSend as e:
                        ambiguous = True
                        self.last_error = str(e)
                        print(f"[GSM] {number}: {e} -> not retrying (might have been sent)")
                    except Exception as e:
                        self.last_error = str(e)
                        print(f"[GSM] attempt {attempt + 1}/3 failed for {number} (part {done + 1}/{len(plan)}): {e}")
                        self.close()
                if sent:
                    self.last_sent = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.last_error = ""
                    print(f"[GSM] SMS sent to {number} ({len(plan)} part{'s' if len(plan) > 1 else ''}) "
                          f"| priority {prio} | queue wait {t_start - t_enq:.1f}s | send {time.time() - t_start:.1f}s")
                    break
                if ambiguous:
                    break
                time.sleep(1)
            self.busy = False
            self.q.task_done()

    def refresh_network(self):
        """Keep network-registration status fresh while idle, so sending never waits on a CREG check."""
        try:
            ok, resp = self._cmd("AT+CREG?", timeout=2)
            if "+CREG:" in resp and (",1" in resp or ",5" in resp):
                self.status = "ready"
                self.last_reg_ok = time.time()
        except Exception:
            pass

    def diagnose(self):
        with self.io_lock:
            if not self.dev or not self.dev.is_open:
                self.connect()
            if not self.dev:
                return {"connected": False, "error": self.last_error, "ports_seen": self._candidate_ports()}
            out = {"connected": True, "port": self.port, "status": self.status}
            checks = (("at", "AT"), ("sim", "AT+CPIN?"), ("signal", "AT+CSQ"),
                      ("network", "AT+CREG?"), ("operator", "AT+COPS?"),
                      ("smsc", "AT+CSCA?"), ("own_number", "AT+CNUM"))
            for label, cmd in checks:
                try:
                    _, resp = self._cmd(cmd, timeout=3)
                    out[label] = resp.strip()
                except Exception as e:
                    out[label] = f"error: {e}"
            return out

    def read_incoming(self):
        if not self.dev or not self.dev.is_open:
            return []
        try:
            self.dev.reset_input_buffer()
            self.dev.write(b'AT+CMGL="REC UNREAD"\r')
            msgs = parse_cmgl(self._read_until((b"OK", b"ERROR"), 5))
            for m in msgs:
                self._cmd(f"AT+CMGD={m['index']}")
            return msgs
        except Exception as e:
            self.last_error = f"read sms error: {e}"
            self.close()
            return []

    def send_sms(self, text, numbers=None, priority=1):
        text = sanitize_sms(text)
        sim_num = normalize_ph_number(GSM_SIM_NUMBER)
        now = time.time()
        for k in [k for k, t in self.recent.items() if now - t > 120]:
            self.recent.pop(k, None)
        for n in (numbers or SMS_RECIPIENTS):
            num = normalize_ph_number(n)
            if num == sim_num:
                print(f"[GSM] Skipped {n} - this is the modem's own SIM number")
            elif num and "X" not in num.upper():
                key = (num, text)
                if now - self.recent.get(key, 0.0) < self.DEDUPE_SEC:
                    print(f"[GSM] Duplicate SMS to {num} ignored")
                    continue
                self.recent[key] = now
                self.q.put((priority, next(self._seq), time.time(), num, text))
            else:
                print(f"[GSM] Skipped invalid recipient '{n}' - check SMS_RECIPIENTS")

gsm = GSMModem()

def _gsm_boot():
    try:
        with gsm.io_lock:
            gsm.connect()
        print(f"[GSM] boot status: {gsm.status} {gsm.last_error}")
    except Exception as e:
        gsm.last_error = str(e)
        print(f"[GSM] boot connect failed: {e}")
    finally:
        gsm_boot_done.set()

    if SMS_ON_STARTUP:
        end = time.time() + STARTUP_GPS_WAIT_SEC
        while time.time() < end:
            s = gps.snapshot()
            if s["fix"]:
                break
            if gps.first_scan_done.is_set() and not gps.port:
                break
            time.sleep(0.5)
        gsm.send_sms(build_online_sms(), priority=2)

threading.Thread(target=_gsm_boot, daemon=True).start()

# ── SMS Message Templates ──────────────────────────────
SMS_DIVIDER = "-" * 34

def sms_now():
    d = datetime.now()
    return f"{d:%A, %B} {d.day}, {d:%Y} | {d:%I:%M:%S %p}"

def compose_sms(title, fields, intro=None, footer=None):
    lines = [title, SMS_DIVIDER]
    if intro:
        lines += [intro, ""]
    lines += [f"{k}: {v}" for k, v in fields]
    if footer:
        lines.append(footer)
    return "\n".join(lines)

def _driver_fields(name, plate):
    return [("Driver", (name or "Unknown").strip()),
            ("Vehicle Plate", (plate or "N/A").strip()),
            ("Date & Time", sms_now())]

def build_online_sms():
    return compose_sms("SYSTEM NOTIFICATION: ONLINE",
                       [("Date & Time", sms_now()),
                        ("Location", gps.location_text()),
                        ("Status", "System Ready for Verification")],
                       intro="Fleet Safety & Monitoring System Has Been Activated.")

def build_timein_sms(name, plate):
    return compose_sms("DRIVER VERIFICATION LOG (TIME-IN)",
                       _driver_fields(name, plate) +
                       [("Location", gps.location_text()),
                        ("Status", "Face Scan Verified / Awaiting Breathalyzer")])

def build_breath_sms(name, plate, peak, passed):
    if passed:
        return compose_sms("ALCOHOL TEST RESULT: CLEAR (PASSED)",
                           _driver_fields(name, plate) +
                           [("Alcohol Level", f"{peak} PPM (Negative)"),
                            ("Location", gps.location_text()),
                            ("Status", "Ignition Authorized / Engine Enabled")])
    return compose_sms("CRITICAL ALERT: ALCOHOL DETECTED",
                       _driver_fields(name, plate) +
                       [("Alcohol Level", f"{peak} PPM (Positive)"),
                        ("Location", gps.location_text()),
                        ("Status", "ENGINE LOCKED / Ignition Denied")],
                       footer=f"Engine stays locked until an admin resets it (reply RESET). Retest allowed {LOCKOUT_COOLDOWN_SEC}s after the reset.")

def build_alcohol_sms(name, plate, peak):
    return build_breath_sms(name, plate, peak, passed=False)

def build_drowsy_sms(name, plate, alert_count):
    return compose_sms("CRITICAL ALERT: DROWSINESS DETECTED",
                       _driver_fields(name, plate) +
                       [("Drowsy Alerts", alert_count),
                        ("Location", gps.location_text()),
                        ("Status", "Drowsy Driver / Buzzer Activated")])

def build_location_sms(name, plate):
    return compose_sms("DRIVER LOCATION UPDATE",
                       _driver_fields(name, plate) +
                       [("Location", gps.location_text()),
                        ("Status", "Manual Location Request")])

def build_test_sms():
    return compose_sms("SYSTEM TEST MESSAGE",
                       [("Date & Time", sms_now()),
                        ("Location", gps.location_text()),
                        ("Status", "SMS Notification Service Operational")])

def build_unlock_sms():
    return compose_sms("ENGINE UNLOCKED (RESET)",
                       [("Date & Time", sms_now()),
                        ("Location", gps.location_text()),
                        ("Status", f"Interlock Reset by SMS / Retest allowed after {LOCKOUT_COOLDOWN_SEC}s cooldown")])

def build_reset_failed_sms():
    return compose_sms("RESET FAILED",
                       [("Date & Time", sms_now()),
                        ("Status", "No Response from Interlock / Try Again or Reset at Vehicle")])

DRIVER_STATUS_TEXT = {"AWAKE": "Awake", "WATCHING": "Awake (Monitoring)",
                      "WARNING": "Drowsiness Warning", "DROWSY": "DROWSY (Alert Active)",
                      "NO_DRIVER": "No Driver Detected"}

def build_session_end_sms(sm):
    alc = sm.get("alcohol_result") or "N/A"
    if sm.get("alcohol_level") not in ("", None) and alc != "N/A":
        alc = f"{alc} ({sm['alcohol_level']} PPM)"
    return compose_sms("DRIVER SESSION ENDED (TIME-OUT)",
                       [("Driver", (sm.get("name") or "Unknown").strip()),
                        ("Vehicle Plate", (sm.get("plate") or "N/A").strip()),
                        ("Date", sm.get("date", "")),
                        ("Time-In", sm.get("login", "")),
                        ("Time-Out", sm.get("logout", "")),
                        ("Duration", sm.get("duration", "")),
                        ("Drowsy Alerts", sm.get("alert_count", 0)),
                        ("Warnings", sm.get("warning_count", 0)),
                        ("Yawns", sm.get("yawns", 0)),
                        ("Alcohol Test", alc),
                        ("Rating", sm.get("rating", "")),
                        ("Location", gps.location_text()),
                        ("Status", "Trip Completed / Session Logged")])

def build_engine_off_sms(name, plate, status, alert_count):
    dstat = DRIVER_STATUS_TEXT.get(status, status or "Unknown")
    footer = "CAUTION: Ignition was turned off while the driver was drowsy." if status == "DROWSY" else None
    return compose_sms("IGNITION OFF DURING MONITORING",
                       _driver_fields(name, plate) +
                       [("Driver Status", dstat),
                        ("Drowsy Alerts", alert_count),
                        ("Location", gps.location_text()),
                        ("Status", f"Engine Off / {GRACE_SECONDS}s Grace Period Started")],
                       footer=footer)

def build_no_lock_sms():
    return compose_sms("RESET IGNORED",
                       [("Date & Time", sms_now()),
                        ("Status", "No Active Engine Lock")])

# ── Attendance + Grace Period ──────────────────────────
def rating_for(alert_count):
    return "Excellent" if alert_count == 0 else \
           "Good"      if alert_count <= 3 else \
           "Fair"      if alert_count <= 5 else "Poor"

def log_attendance(name, plate, login_dt, logout_dt, alert_count, rating, extra=None):
    secs = max(0, int((logout_dt - login_dt).total_seconds()))
    hours, rem = divmod(secs, 3600)
    mins, sec  = divmod(rem, 60)
    duration   = f"{hours}h {mins:02d}m {sec:02d}s"
    logout_str = logout_dt.strftime(TIME_FMT)
    day_diff = (logout_dt.date() - login_dt.date()).days
    if day_diff > 0:
        logout_str += f" (+{day_diff}d)"
    ex = extra or {}
    append_csv(ATTENDANCE_CSV, TRIPS_HEADER,
               [name, plate, login_dt.strftime(TIME_FMT), logout_str, duration,
                alert_count, rating, login_dt.strftime(DATE_FMT),
                ex.get("warnings", 0), ex.get("yawns", 0),
                ex.get("alcohol_result", "") or "", ex.get("alcohol_level", ""),
                ex.get("lat", ""), ex.get("lon", ""), "COMPLETED"])
    return duration, logout_str

def finalize_session(logout_dt=None):
    with session_lock:
        if session["logged"]:
            return session["summary"]
        if not session["driver_name"] or not session["login_dt"] or not session["started"]:
            return None
        logout_dt   = logout_dt or datetime.now()
        alert_count = len(session["alerts"])
        warn_count  = len(session["warn_events"])
        rating      = rating_for(alert_count)
        duration, logout_str = log_attendance(
            session["driver_name"], session["plate_number"],
            session["login_dt"], logout_dt, alert_count, rating,
            extra={"warnings": warn_count, "yawns": session["yawns"],
                   "alcohol_result": session["alcohol_result"],
                   "alcohol_level": session["alcohol_level"],
                   "lat": session["login_lat"], "lon": session["login_lon"]})
        summary = {
            "name":        session["driver_name"],
            "plate":       session["plate_number"],
            "date":        session["login_dt"].strftime(DATE_FMT),
            "login":       session["login_dt"].strftime(TIME_FMT),
            "logout":      logout_str,
            "duration":    duration,
            "alert_count": alert_count,
            "alerts":      list(session["alerts"]),
            "warning_count": warn_count,
            "warnings":    warn_count,
            "yawns":       session["yawns"],
            "videos":      session["videos"],
            "alcohol_result": session["alcohol_result"],
            "alcohol_level":  session["alcohol_level"],
            "rating":      rating,
        }
        session["summary"] = summary
        session["logged"] = True
        session["engine_off_at"] = None
    send_to_esp32("GRACE_CLEAR")
    if SMS_NOTIFY_SESSION_END and summary:
        try:
            gsm.send_sms(build_session_end_sms(summary))
        except Exception as e:
            print(f"[sms] session-end SMS error: {e}")
    return summary

def record_blocked_session(peak):
    with session_lock:
        if not session["driver_name"] or session["logged"]:
            return
        login_dt = session["login_dt"] or datetime.now()
        summary = {
            "name": session["driver_name"], "plate": session["plate_number"],
            "date": login_dt.strftime(DATE_FMT), "login": login_dt.strftime(TIME_FMT),
            "logout": "-", "duration": "0h 00m 00s",
            "alert_count": 0, "alerts": [], "warning_count": 0, "warnings": 0, "yawns": 0, "videos": 0,
            "alcohol_result": "DETECTED", "alcohol_level": peak, "rating": "N/A",
        }
        append_csv(ATTENDANCE_CSV, TRIPS_HEADER,
                   [session["driver_name"], session["plate_number"], login_dt.strftime(TIME_FMT), "-",
                    "0h 00m 00s", 0, "N/A", login_dt.strftime(DATE_FMT), 0, 0, "DETECTED", peak,
                    session["login_lat"], session["login_lon"], "BLOCKED"])
        session.update(alcohol_result="DETECTED", alcohol_level=peak,
                       logged=True, summary=summary, engine_off_at=None)

def check_grace_expiry():
    try:
        off_at = session["engine_off_at"]
        if off_at and not session["logged"]:
            if (datetime.now() - off_at).total_seconds() >= GRACE_SECONDS + GRACE_BUFFER:
                finalize_session(off_at)
    except Exception as e:
        print(f"Grace check error: {e}")

# ── Vibration Motor (V919) + Speaker Alerts ────────────   # <-- NEW (whole section, more robust sound)
VIB_PIN          = 17      # BCM GPIO17 (physical pin 11)
VIB_ACTIVE_HIGH  = os.environ.get("VIB_ACTIVE_HIGH", "1") == "1"   # If inverted (vibrates when OFF): VIB_ACTIVE_HIGH=0 python3 app.py
VIB_WARN_ON_SEC  = 0.5     # WARNING: vibrate ON duration
VIB_WARN_OFF_SEC = 0.5     # WARNING: vibrate OFF duration
AUDIO_DEVICE     = os.environ.get("ALERTX_AUDIO_DEVICE")   # Optional manual ALSA device, ex. "plughw:2,0". If unset, auto-detected
VOICE_TEXT       = "Warning. Driver is drowsy. Please wake up and stay alert."
VOICE_REPEAT_SEC = 3.0     # How often the voice repeats while drowsy
SOUND_DIR        = os.path.join(BASE_DIR, "sounds")
WARN_WAV         = os.path.join(SOUND_DIR, "warn_beep.wav")
VOICE_WAV        = os.path.join(SOUND_DIR, "drowsy_voice.wav")
CUSTOM_WARN_MP3   = None    # WARNING = loud "beep beep" (generated), no longer alert.mp3
CUSTOM_DROWSY_MP3 = os.path.join(BASE_DIR, "alert.mp3")   # DROWSY = spoken alert (if missing: espeak voice, then beep)

def make_beep_wav(path, beeps=2, freq=2300, beep_sec=0.28, gap_sec=0.14, rate=44100):
    """Loud, sharp 'beep beep' (square-ish wave, with a fade to avoid clicks)."""
    n = int(rate * beep_sec)
    t = np.linspace(0, beep_sec, n, False)
    sine = np.sin(2 * np.pi * freq * t)
    square = np.sign(sine)
    wave_f = 0.55 * square + 0.45 * sine           # sharper than a pure sine, more attention-grabbing
    fade = int(rate * 0.008)
    env = np.ones(n)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    tone = (wave_f * env * 0.95 * 32767).astype(np.int16)
    gap = np.zeros(int(rate * gap_sec), dtype=np.int16)
    data = np.concatenate([np.concatenate([tone, gap]) for _ in range(beeps)])
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(data.tobytes())

def mp3_to_wav(src, dst):
    """Converts mp3 -> wav. Returns the wav path, or None on failure."""
    try:
        if not src or not os.path.isfile(src):
            print(f"[audio] file not found: {src}")
            return None
        if os.path.isfile(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
            return dst
        ff = shutil.which("ffmpeg")
        if not ff:
            print("[audio] ffmpeg not installed (sudo apt install ffmpeg)")
            return None
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", src,
                        "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le", dst],
                       check=True, timeout=60)
        print(f"[audio] converted {os.path.basename(src)} -> {os.path.basename(dst)}")
        return dst
    except Exception as e:
        print(f"[audio] mp3 convert failed: {e}")
        return None

def aplay_list():
    try:
        r = subprocess.run(["aplay", "-l"], capture_output=True, text=True, timeout=5)
        return (r.stdout or "") + (r.stderr or "")
    except Exception as e:
        return f"error: {e}"

def detect_audio_device():
    """Find the plugged-in speaker (USB soundbar/USB sound card). Skips HDMI."""
    if AUDIO_DEVICE:
        return AUDIO_DEVICE
    for m in re.finditer(r"card (\d+): (\S+) \[([^\]]*)\], device (\d+):", aplay_list()):
        _, cid, desc, dev = m.groups()
        if "hdmi" in (cid + desc).lower():
            continue
        return f"plughw:CARD={cid},DEV={dev}"
    return None

def audio_env():
    """Makes PipeWire/PulseAudio work even when the app is run from an SSH / VS Code terminal."""
    env = os.environ.copy()
    try:
        uid = int(os.environ.get("SUDO_UID", os.getuid()))
    except Exception:
        uid = os.getuid()
    rd = f"/run/user/{uid}"
    if os.path.isdir(rd):
        env.setdefault("XDG_RUNTIME_DIR", rd)
        if os.path.exists(f"{rd}/pulse/native"):
            env.setdefault("PULSE_SERVER", f"unix:{rd}/pulse/native")
    return env

class AlertOutput:
    """WARNING = pulsing vibration + sound on the speaker | DROWSY = continuous vibration + voice on the speaker."""
    def __init__(self):
        self.mode = "off"          # "off" / "warn" / "drowsy"
        self.hold_until = 0.0      # For manual testing (/api/test_alert)
        self.vib = None
        self.audio_dev = None
        self.dev_checked = 0.0
        self.gen = 0               # Cancel token for the currently playing sound
        self.playing = False
        self.proc = None
        self.good = None           # Name of the player that worked
        try:
            self.vib = OutputDevice(VIB_PIN, active_high=VIB_ACTIVE_HIGH, initial_value=False)
            print(f"[vib] V919 ready on GPIO{VIB_PIN}")
        except Exception as e:
            print(f"[vib] cannot init GPIO{VIB_PIN}: {e}")
        os.makedirs(SOUND_DIR, exist_ok=True)
        try:
            make_beep_wav(WARN_WAV)
        except Exception as e:
            print(f"[audio] cannot create beep: {e}")
        self.voice_ok = self._make_voice()
        _w = mp3_to_wav(CUSTOM_WARN_MP3, os.path.join(SOUND_DIR, "custom_warn.wav")) if CUSTOM_WARN_MP3 else None
        self.warn_wav = _w or WARN_WAV
        _d = mp3_to_wav(CUSTOM_DROWSY_MP3, os.path.join(SOUND_DIR, "custom_drowsy.wav")) if CUSTOM_DROWSY_MP3 else None
        self.drowsy_wav = _d or (VOICE_WAV if self.voice_ok else self.warn_wav)
        self._refresh_audio_dev()
        avail = [n for n in ("pw-play", "paplay", "aplay", "ffplay") if shutil.which(n)]
        print(f"[audio] players available: {', '.join(avail) or 'NONE'}")
        threading.Thread(target=self._run, daemon=True).start()

    def _refresh_audio_dev(self):
        self.dev_checked = time.time()
        dev = detect_audio_device()
        if dev != self.audio_dev:
            self.audio_dev = dev
            print(f"[audio] USB/ALSA device: {dev or 'none (the default output will be used)'}")

    def _make_voice(self):
        exe = shutil.which("espeak-ng") or shutil.which("espeak")
        if not exe:
            print("[audio] espeak-ng not installed -> only alert.mp3/beep will be used for drowsy")
            return False
        try:
            subprocess.run([exe, "-s", "140", "-w", VOICE_WAV, VOICE_TEXT], check=True, timeout=20)
            return os.path.isfile(VOICE_WAV)
        except Exception as e:
            print(f"[audio] voice generation failed: {e}")
            return False

    def _vib(self, on):
        if self.vib is None:
            return
        try:
            self.vib.on() if on else self.vib.off()
        except Exception as e:
            print(f"[vib] error: {e}")

    def _players(self, path):
        """List of ways to play the sound, tried one by one until one works."""
        cmds = []
        if AUDIO_DEVICE:
            cmds.append(("aplay(manual)", ["aplay", "-q", "-D", AUDIO_DEVICE, path]))
        if shutil.which("pw-play"):
            cmds.append(("pw-play", ["pw-play", path]))
        if shutil.which("paplay"):
            cmds.append(("paplay", ["paplay", path]))
        if self.audio_dev and not AUDIO_DEVICE:
            cmds.append(("aplay(usb)", ["aplay", "-q", "-D", self.audio_dev, path]))
        cmds.append(("aplay(default)", ["aplay", "-q", path]))
        if shutil.which("ffplay"):
            cmds.append(("ffplay", ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path]))
        if self.good:
            cmds.sort(key=lambda c: 0 if c[0] == self.good else 1)
        return cmds

    def _run_chain(self, path, gen, report=None):
        env = audio_env()
        ok = False
        for name, cmd in self._players(path):
            if gen != self.gen:
                break
            try:
                p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env)
                self.proc = p
                try:
                    _, err = p.communicate(timeout=30)
                except subprocess.TimeoutExpired:
                    p.kill()
                    _, err = p.communicate()
                rc = p.returncode
                msg = (err or b"").decode("utf-8", "ignore").strip()[-200:]
            except Exception as e:
                rc, msg = -1, str(e)
            if report is not None:
                report.append({"player": name, "cmd": " ".join(cmd), "returncode": rc, "stderr": msg})
            if gen != self.gen:
                break
            if rc == 0:
                if self.good != name:
                    print(f"[audio] OK via {name}")
                self.good = name
                ok = True
                break
            print(f"[audio] {name} failed (rc={rc}): {msg}")
        if not ok and gen == self.gen:
            print("[audio] NO player worked - open /api/test_sound to see why")
        return ok

    def _play(self, path):
        """Returns True if a new sound was started."""
        if self.playing:
            return False
        if not self.audio_dev and time.time() - self.dev_checked > 10:
            self._refresh_audio_dev()   # the speaker may have just been plugged in
        gen = self.gen
        self.playing = True

        def work():
            try:
                self._run_chain(path, gen)
            except Exception as e:
                print(f"[audio] error: {e}")
            finally:
                self.playing = False

        threading.Thread(target=work, daemon=True).start()
        return True

    def _stop_sound(self):
        self.gen += 1
        p = self.proc
        if p and p.poll() is None:
            try:
                p.terminate()
            except Exception:
                pass

    def set_mode(self, mode, force=False, hold=0):
        if not force and time.time() < self.hold_until:
            return   # a manual test is running
        if force:
            self.hold_until = time.time() + hold
        self.mode = mode

    def _run(self):
        prev = "off"
        last_beep = last_voice = vib_t = 0.0
        vib_state = False
        while True:
            try:
                mode, now = self.mode, time.time()
                if mode != prev:
                    prev = mode
                    last_beep = last_voice = vib_t = 0.0
                    vib_state = False
                    self._stop_sound()
                    self._vib(False)
                if mode == "warn":
                    if now - vib_t >= (VIB_WARN_ON_SEC if vib_state else VIB_WARN_OFF_SEC):
                        vib_state = not vib_state
                        vib_t = now
                        self._vib(vib_state)
                    if now - last_beep >= WARN_BEEP_REPEAT:
                        if self._play(self.warn_wav):
                            last_beep = now
                elif mode == "drowsy":
                    if not vib_state:
                        vib_state = True
                        self._vib(True)          # continuous, never off
                    if now - last_voice >= VOICE_REPEAT_SEC:
                        if self._play(self.drowsy_wav):
                            last_voice = now
            except Exception as e:
                print(f"[alert-out] error: {e}")
            time.sleep(0.05)

alert_out = AlertOutput()

# ── Drowsiness Detection ───────────────────────────────
class StateTimer:
    def __init__(self, gap):
        self.gap = gap
        self.reset()

    def reset(self):
        self.since = None
        self.last = 0.0

    def update(self, on, now, gap=None):
        gap = self.gap if gap is None else gap
        if on:
            if self.since is None:
                self.since = now
            self.last = now
        elif self.since is not None and now - self.last > gap:
            self.reset()

    def span(self):
        return (self.last - self.since) if self.since is not None else 0.0

    def live(self, now, within=0.7):
        return self.since is not None and (now - self.last) <= within

def held(timer, now):
    return timer.span() if timer.live(now) else 0.0

def analyze(detected):
    """Lower thresholds so detection works even from far away; accuracy is ensured by time + ratio confirmation."""
    best = {}
    for name, conf in detected:
        if name in KNOWN_CLASSES and conf > best.get(name, 0.0):
            best[name] = conf
    g = lambda *names: max((best.get(n, 0.0) for n in names), default=0.0)
    closed_c = g('eye_closed_L', 'eye_closed_R', 'eye_close_L')
    open_c   = g('eye_open_L', 'eye_open_R')
    eyes_closed = closed_c >= 0.30 and closed_c > open_c
    eyes_open   = open_c >= 0.30 and open_c >= closed_c
    mouth_c = g('mouth_open')
    yawning = mouth_c >= 0.35 and mouth_c > g('mouth_closed')
    down_c = g('head_down')
    head_down = down_c >= 0.40 and down_c > g('head_normal')
    tilt_c = g('head_tilt_L', 'head_tilt_R')
    head_tilt = tilt_c >= 0.40 and tilt_c > g('head_normal')
    d_c = g('drowsy')
    drowsy_face = d_c >= 0.50 and d_c > g('awake') and not yawning
    sleeping = (eyes_closed or head_down or drowsy_face) and not eyes_open
    return {
        "has_driver":  bool(best) and max(best.values()) >= DETECT_CONF,
        "eyes_closed": eyes_closed, "eyes_open": eyes_open,
        "yawning": yawning, "head_down": head_down, "head_tilt": head_tilt,
        "sleeping": sleeping,
    }

HIST_IDX = {"sleeping": 1, "closed": 2, "down": 3, "tilt": 4, "yawn": 5, "open": 6}

class DriverMonitor:
    def __init__(self):
        self.lock = threading.RLock()
        self.active = False
        self._reset()
        self.snap = self._blank("INACTIVE")

    def _blank(self, status, condition=""):
        return {"status": status, "condition": condition, "signs": [],
                "drowsy_for": 0.0, "detected": [], "updated": time.time()}

    def _reset(self):
        self.sleep  = StateTimer(SLEEP_GAP_SEC)
        self.closed = StateTimer(SLEEP_GAP_SEC)
        self.down   = StateTimer(SLEEP_GAP_SEC)
        self.tilt   = StateTimer(SLEEP_GAP_SEC)
        self.yawn   = StateTimer(1.2)     # warning timer (bridges brief mouth-closing during a yawn)
        self.ystat  = StateTimer(0.6)     # stats timer (yawn count)
        self.opened = StateTimer(0.4)     # eyes open
        self.hist = deque()
        self.yawn_counted = False
        self.drowsy = False
        self.awake_since = None
        self.last_ping = 0.0
        self.last_seen = 0.0
        self.warn_active = set()
        self.last_warn_beep = 0.0

    def _clear_signs(self):
        for t in (self.sleep, self.closed, self.down, self.tilt, self.yawn):
            t.reset()
        self.hist.clear()

    def _ratio(self, key, window, now):
        i = HIST_IDX[key]
        tot = hit = 0
        for rec in self.hist:
            if now - rec[0] <= window:
                tot += 1
                if rec[i]:
                    hit += 1
        return (hit / tot if tot else 0.0), tot

    def _confirmed(self, key, window, need, now):
        r, tot = self._ratio(key, window, now)
        return tot >= HIST_MIN_FRAMES and r >= need

    def snapshot(self):
        with self.lock:
            return dict(self.snap)

    def deactivate(self, status="INACTIVE"):
        alert_out.set_mode("off")
        with self.lock:
            if self.active or self.drowsy:
                send_to_esp32("DROWSY_OFF")
                recorder.stop()
                self._reset()
                self.active = False
            self.snap = self._blank(status)

    def _start_episode(self, now, detail=""):
        global last_drowsy_sms
        self.drowsy = True
        self.awake_since = None
        self.last_ping = now
        self.warn_active = set()
        with session_lock:
            session["alerts"].append(datetime.now().strftime(TIME_FMT))
            alert_count = len(session["alerts"])
        log_drowsy("Drowsy Alert", detail or "Continuous drowsiness")
        recorder.start(session["driver_name"], session["plate_number"], detail or "Continuous drowsiness")
        send_to_esp32("DROWSY_ON")
        if (now - last_drowsy_sms) >= SMS_DROWSY_COOLDOWN_SEC:
            gsm.send_sms(build_drowsy_sms(session["driver_name"], session["plate_number"], alert_count), priority=0)
            last_drowsy_sms = now

    def _end_episode(self):
        self.drowsy = False
        self.awake_since = None
        self._clear_signs()
        send_to_esp32("DROWSY_OFF")
        recorder.stop()

    def update(self, sig, now, names):
        with self.lock:
            self.active = True
            if sig["has_driver"]:
                self.last_seen = now
            present = (now - self.last_seen) < NO_DRIVER_SEC

            # Eyes open = AWAKE only. No warning, and all counters are reset.
            self.opened.update(sig["eyes_open"], now)
            open_for = held(self.opened, now)
            awake_now = open_for >= AWAKE_CLEAR_SEC
            if awake_now and not self.drowsy:
                self._clear_signs()

            feed = dict(sig)
            if awake_now:
                feed.update(eyes_closed=False, head_down=False, head_tilt=False,
                            yawning=False, sleeping=False)

            gap = AWAKE_CLEAR_SEC if sig["eyes_open"] else None
            self.sleep.update(feed["sleeping"], now, gap)
            self.closed.update(feed["eyes_closed"], now, gap)
            self.down.update(feed["head_down"], now, gap)
            self.tilt.update(feed["head_tilt"], now, gap)
            self.yawn.update(feed["yawning"], now, gap)
            self.ystat.update(sig["yawning"], now)

            self.hist.append((now, feed["sleeping"], feed["eyes_closed"], feed["head_down"],
                              feed["head_tilt"], feed["yawning"], sig["eyes_open"]))
            cutoff = now - (DROWSY_SECONDS + 3.0)
            while self.hist and self.hist[0][0] < cutoff:
                self.hist.popleft()

            sleep_for  = held(self.sleep, now)
            closed_for = held(self.closed, now)
            down_for   = held(self.down, now)
            tilt_for   = held(self.tilt, now)
            yawn_for   = held(self.yawn, now)
            ystat_for  = held(self.ystat, now)

            if ystat_for >= YAWN_COUNT_SEC and not self.yawn_counted:
                self.yawn_counted = True
                with session_lock:
                    session["yawns"] += 1
            if self.ystat.since is None:
                self.yawn_counted = False

            recent_open = any(r[6] and now - r[0] <= 1.0 for r in self.hist)

            if not self.drowsy:
                if (present and feed["sleeping"] and sleep_for >= DROWSY_SECONDS and not recent_open
                        and self._confirmed("sleeping", DROWSY_SECONDS, DROWSY_CONFIRM_RATIO, now)):
                    parts = [n for n, v in (("Eyes closed", closed_for), ("Head down", down_for)) if v >= 0.5]
                    self._start_episode(now, " + ".join(parts) or "Drowsy face")
            else:
                if now - self.last_ping >= 1.0:
                    self.last_ping = now
                    send_to_esp32("DROWSY_ON")
                if sig["sleeping"]:
                    self.awake_since = None
                elif self.awake_since is None:
                    self.awake_since = now
                eyes_ok = open_for >= RECOVER_EYES_SEC
                quiet = self.awake_since is not None and (now - self.awake_since) >= RECOVER_SEC
                recovered = present and (eyes_ok or quiet)
                lost = (not present) and (now - self.last_seen) > FACELOST_MAX_SEC
                if recovered or lost:
                    self._end_episode()
                    sleep_for = closed_for = down_for = tilt_for = yawn_for = 0.0

            warnings = []
            if not self.drowsy and present and not awake_now:
                ok = lambda key, held_for, secs: held_for >= secs and self._confirmed(key, secs, WARN_CONFIRM_RATIO, now)
                if ok("closed", closed_for, WARN_EYES_SEC): warnings.append("Eyes closed")
                if ok("down",   down_for,   WARN_HEAD_SEC): warnings.append("Head down")
                if ok("yawn",   yawn_for,   WARN_YAWN_SEC): warnings.append("Yawning")
                if ok("tilt",   tilt_for,   WARN_HEAD_SEC): warnings.append("Head tilt")
                if not warnings and ok("sleeping", sleep_for, WARN_SECONDS):
                    warnings.append("Drowsy signs")

            new = [w for w in warnings if w not in self.warn_active]
            if new:
                stamp = datetime.now().strftime(TIME_FMT)
                with session_lock:
                    for w in new:
                        session["warn_events"].append({"time": stamp, "type": w})
                for w in new:
                    log_drowsy("Warning", w)

            if warnings and now - self.last_warn_beep >= WARN_BEEP_REPEAT:
                self.last_warn_beep = now
                send_to_esp32("WARN")
            self.warn_active = set(warnings)
            alert_out.set_mode("drowsy" if self.drowsy else ("warn" if warnings else "off"))

            signs = []
            if not awake_now:
                if closed_for >= 0.5: signs.append("Eyes closed")
                if down_for   >= 0.5: signs.append("Head down")
                if yawn_for   >= 0.5: signs.append("Yawning")
                if tilt_for   >= 0.5: signs.append("Head tilt")
                if not signs and sleep_for >= 0.5: signs.append("Drowsy")

            if self.drowsy:
                status, condition = "DROWSY", (" + ".join(signs) or "Drowsy")
            elif not present:
                status, condition, signs = "NO_DRIVER", "No driver", []
            elif warnings:
                status, condition = "WARNING", " + ".join(warnings)
            elif sleep_for >= 0.8 and not awake_now:
                status, condition = "WATCHING", (" + ".join(signs) or "Drowsy signs")
            else:
                status, condition, signs = "AWAKE", "Awake", []

            self.snap = {"status": status, "condition": condition, "signs": signs,
                         "drowsy_for": round(sleep_for, 1), "detected": names,
                         "updated": time.time()}
            session["current_status"] = status

driver_monitor = DriverMonitor()

EYE_CLASSES = {'eye_closed_L', 'eye_closed_R', 'eye_close_L', 'eye_open_L', 'eye_open_R'}

def _infer(img):
    out = []
    for r in model(img, imgsz=640, conf=DETECT_CONF, verbose=False):
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            out.append((model.names[int(box.cls[0])], float(box.conf[0]), (x1, y1, x2, y2)))
    return out

def _has_eyes(dets):
    return any(n in EYE_CLASSES and c >= 0.30 for n, c, _ in dets)

def _zoom_crop(frame, dets):
    """Crop around the detected face (or the center if none) and upscale, so the eyes/mouth appear larger."""
    h, w = frame.shape[:2]
    if dets:
        x1 = min(d[2][0] for d in dets); y1 = min(d[2][1] for d in dets)
        x2 = max(d[2][2] for d in dets); y2 = max(d[2][3] for d in dets)
        bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        cw = max(bw * (1 + 2 * ZOOM_PAD), bh * (1 + 2 * ZOOM_PAD) * w / h, w * 0.35)
    else:
        cx, cy = w / 2, h / 2
        cw = w * ZOOM_CENTER_FRAC
    cw = min(cw, w)
    ch = cw * h / w
    if cw >= w * 0.95:
        return None
    x0 = min(max(cx - cw / 2, 0), w - cw)
    y0 = min(max(cy - ch / 2, 0), h - ch)
    crop = frame[int(y0):int(y0 + ch), int(x0):int(x0 + cw)]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC)

def detector_loop():
    while True:
        try:
            if is_system_locked():
                driver_monitor.deactivate("LOCKED")
                time.sleep(0.3)
                continue
            if not monitoring_on():
                driver_monitor.deactivate("INACTIVE")
                time.sleep(0.3)
                continue
            frame = camera.read()
            if frame is None:
                time.sleep(0.1)
                continue
            dets = _infer(frame)
            if ZOOM_ENABLED and not _has_eyes(dets):
                crop = _zoom_crop(frame, dets)
                if crop is not None:
                    dets = dets + _infer(crop)
            named = [(n, c) for n, c, _ in dets]
            driver_monitor.update(analyze(named), time.time(), [n for n, _ in named])
            time.sleep(0.02)
        except Exception as e:
            print(f"[detect] error: {e}")
            time.sleep(0.5)

# ── ESP32 (Bluetooth) Reader ───────────────────────────
TEST_STATES = ("WAIT_FACE", "WARMUP", "TESTING")

def handle_esp32_packet(data, prev_state):
    global esp32_data, esp32_last_seen, last_engine_off_sms
    esp32_data = data
    esp32_last_seen = time.time()
    state = str(data.get("state", "")).upper()
    val   = data.get("val", 0)
    v     = _num(val, float, 0.0) or 0.0
    if state in ("WARMUP", "TESTING", "LOCKOUT"):
        breath_session["max"] = max(breath_session.get("max", 0), v)
    peak = int(max(breath_session.get("max", 0), v))

    if state == "LOCKOUT" and prev_state != "LOCKOUT":
        if not get_persisted_lock():
            set_system_lock(True, f"Alcohol level: {peak}",
                            driver=session["driver_name"],
                            plate=session["plate_number"],
                            level=peak)
            if prev_state in TEST_STATES:
                log_breath("FAILED - ENGINE LOCKED", level=peak, note="Alcohol limit exceeded")
                gsm.send_sms(build_breath_sms(session["driver_name"], session["plate_number"], peak, passed=False), priority=0)
                record_blocked_session(peak)
            else:
                log_breath("ENGINE LOCKED", level=peak, note="Lock restored from interlock")
            breath_session["max"] = 0

    if state == "PASSED" and prev_state in TEST_STATES:
        log_breath("PASSED", level=peak, note="Within allowed limit")
        session["alcohol_result"] = "CLEAR"
        session["alcohol_level"] = peak
        gsm.send_sms(build_breath_sms(session["driver_name"], session["plate_number"], peak, passed=True), priority=1)
        breath_session["max"] = 0

    if state == "PASSED":
        if session["driver_name"] and not session["logged"]:
            session["started"] = True
        session["engine_off_at"] = None
    elif prev_state == "PASSED" and state == "IDLE":
        if session["driver_name"] and session["started"] and not session["logged"]:
            first_off = session["engine_off_at"] is None
            status_at_off = session.get("current_status", "AWAKE")
            session["engine_off_at"] = datetime.now()
            now_t = time.time()
            if SMS_NOTIFY_ENGINE_OFF and first_off and now_t - last_engine_off_sms >= SMS_ENGINE_OFF_COOLDOWN_SEC:
                last_engine_off_sms = now_t
                gsm.send_sms(build_engine_off_sms(session["driver_name"], session["plate_number"],
                                                  status_at_off, len(session["alerts"])),
                              priority=(0 if status_at_off == "DROWSY" else 1))
    elif state == "WAIT_FACE" and session["engine_off_at"] and not session["logged"]:
        finalize_session(session["engine_off_at"])

    if breath_session["awaiting"]:
        if state == "PASSED" and not get_persisted_lock():
            breath_session.update(result="passed", peak=val, awaiting=False)
        elif state == "LOCKOUT" or get_persisted_lock():
            breath_session.update(result="locked", peak=val, awaiting=False)
    return state

def read_esp32_bt():
    global ser
    connect_bluetooth()
    prev_state = ""
    while True:
        if ser and ser.is_open:
            try:
                while ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if not line:
                        continue
                    if line.startswith("SYSTEM RESET"):
                        if get_persisted_lock() or prev_state == "LOCKOUT":
                            recent = time.time() - lock_ctl["source_at"] < 10
                            src = lock_ctl["source"] if recent else "RFID admin card"
                            clear_persisted_lock(src)
                            esp32_data["state"] = "IDLE"
                        continue
                    if line.startswith('{') and line.endswith('}'):
                        try:
                            packet = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        try:
                            prev_state = handle_esp32_packet(packet, prev_state)
                        except Exception as e:
                            print(f"[bt] packet error: {e}")
            except Exception as e:
                print(f"Bluetooth read error: {e}")
                try:
                    ser.close()
                except Exception:
                    pass
                ser = None
        else:
            time.sleep(2)
            connect_bluetooth()
        enforce_lock()
        check_grace_expiry()
        time.sleep(0.05)

# ── Database Helpers ───────────────────────────────────
def load_drivers():
    if os.path.exists(DRIVER_DB_FILE):
        with open(DRIVER_DB_FILE, "rb") as f:
            db = pickle.load(f)
            if "plates" not in db:
                db["plates"] = ["N/A"] * len(db["names"])
            return db
    return {"encodings": [], "names": [], "plates": []}

def save_drivers(db):
    with open(DRIVER_DB_FILE, "wb") as f:
        pickle.dump(db, f)

def save_driver_info(name, plate_number):
    file_exists = os.path.isfile(DRIVER_INFO_CSV)
    with open(DRIVER_INFO_CSV, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Name", "Plate Number", "Registered Date"])
        writer.writerow([name, plate_number, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

def sync_new_drivers():
    import face_recognition
    os.makedirs(NEW_DRIVERS_DIR, exist_ok=True)
    os.makedirs(DRIVER_PHOTOS, exist_ok=True)
    db = load_drivers()
    added   = []
    skipped = []
    for filename in os.listdir(NEW_DRIVERS_DIR):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        basename = os.path.splitext(filename)[0]
        if "__" not in basename:
            skipped.append({"file": filename, "reason": "Filename must look like Name__Plate.jpg"})
            continue
        name_part, plate_part = basename.split("__", 1)
        name  = name_part.replace("_", " ").strip()
        plate = plate_part.strip()
        if not name or not plate:
            skipped.append({"file": filename, "reason": "Missing name or plate in filename"})
            continue
        filepath = os.path.join(NEW_DRIVERS_DIR, filename)
        try:
            pil_img = PILImage.open(filepath)
            pil_img = ImageOps.exif_transpose(pil_img)
            pil_img = pil_img.convert("RGB")
            MAX_DIMENSION = 1200
            width, height = pil_img.size
            if max(width, height) > MAX_DIMENSION:
                scale = MAX_DIMENSION / max(width, height)
                new_size = (int(width * scale), int(height * scale))
                pil_img = pil_img.resize(new_size, PILImage.LANCZOS)
            image = np.array(pil_img, dtype=np.uint8)
            image = np.ascontiguousarray(image)
            face_locations = face_recognition.face_locations(image, model="hog")
        except Exception as e:
            skipped.append({"file": filename, "reason": f"Format Error: {e}"})
            continue
        if len(face_locations) != 1:
            skipped.append({"file": filename, "reason": f"Expected exactly 1 face, found {len(face_locations)}"})
            continue
        encoding = face_recognition.face_encodings(image, face_locations)[0]
        db["encodings"].append(encoding)
        db["names"].append(name)
        db["plates"].append(plate)
        dest_path = os.path.join(DRIVER_PHOTOS, f"{name}.jpg")
        os.replace(filepath, dest_path)
        save_driver_info(name, plate)
        added.append({"name": name, "plate": plate})
    save_drivers(db)
    return added, skipped

# ── GPS UI Page ────────────────────────────────────────
GPS_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vehicle Location</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
  html,body{margin:0;height:100%;background:#0b1220;color:#e5e7eb;font-family:system-ui,sans-serif}
  .wrap{display:flex;flex-direction:column;height:100%}
  header{padding:10px 14px;display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #1f2a44}
  .badge{padding:3px 10px;border-radius:14px;font-size:12px;font-weight:600}
  .ok{background:rgba(52,211,153,.18);color:#34d399}.warn{background:rgba(251,191,36,.18);color:#fbbf24}.bad{background:rgba(248,113,113,.18);color:#f87171}
  #map{flex:1;min-height:200px;background:#111a33}
  #nomap{display:none;padding:20px;text-align:center;color:#94a3b8}
  .panel{padding:10px 14px;display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px;border-top:1px solid #1f2a44}
  .card{background:#111a33;border-radius:10px;padding:8px 10px}.lbl{font-size:10px;color:#94a3b8}.val{font-size:15px;font-weight:700}
  .actions{padding:10px 14px;display:flex;gap:10px;flex-wrap:wrap}
  button,a.btn{flex:1;min-height:44px;border:0;border-radius:10px;font-weight:700;font-size:14px;cursor:pointer;text-align:center;text-decoration:none;line-height:44px;padding:0 14px}
  #send{background:#ef4444;color:#fff}a.btn{background:#1f2a44;color:#e5e7eb}
  #msg{padding:0 14px 12px;font-size:12px;color:#94a3b8;min-height:16px}
</style></head><body><div class="wrap">
  <header><b>Vehicle Location</b><span id="badge" class="badge bad">No GPS</span></header>
  <div id="map"></div>
  <div id="nomap">Map needs internet. Use "Open in Google Maps" or the coordinates below.</div>
  <div class="panel">
    <div class="card"><div class="lbl">Latitude</div><div class="val" id="lat">-</div></div>
    <div class="card"><div class="lbl">Longitude</div><div class="val" id="lon">-</div></div>
    <div class="card"><div class="lbl">Speed</div><div class="val" id="spd">-</div></div>
    <div class="card"><div class="lbl">Satellites</div><div class="val" id="sat">-</div></div>
  </div>
  <div class="actions">
    <button id="send">Send location by SMS</button>
    <a class="btn" id="open" href="#" target="_blank" rel="noopener">Open in Google Maps</a>
  </div>
  <div id="msg"></div>
</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
  var hasMap = (typeof L !== 'undefined'), map, marker, first = true;
  if (hasMap) { map = L.map('map').setView([12.8797,121.7740], 5);
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {maxZoom:19, attribution:'&copy; OpenStreetMap'}).addTo(map);
  } else { document.getElementById('map').style.display='none'; document.getElementById('nomap').style.display='block'; }
  var recipients = [];
  async function refresh(){
    try{
      var d = await (await fetch('/api/gps', {cache:'no-store'})).json();
      var b = document.getElementById('badge');
      if (d.status === 'fix'){ b.className='badge ok'; b.textContent='GPS fix'; }
      else if (d.status === 'no_fix'){ b.className='badge warn'; b.textContent = d.has_position ? 'No fix - last known' : 'Searching satellites...'; }
      else { b.className='badge bad'; b.textContent='GPS not connected'; }
      recipients = d.sms_recipients || [];
      document.getElementById('sat').textContent = d.sats || 0;
      document.getElementById('spd').textContent = d.speed_kmh != null ? d.speed_kmh + ' km/h' : '-';
      if (d.has_position){
        document.getElementById('lat').textContent = d.lat.toFixed(5);
        document.getElementById('lon').textContent = d.lon.toFixed(5);
        document.getElementById('open').href = d.maps_url;
        if (hasMap){
          var p = [d.lat, d.lon];
          if (!marker) marker = L.marker(p).addTo(map); else marker.setLatLng(p);
          if (first){ map.setView(p, 17); first = false; } else { map.panTo(p); }
        }
      }
    }catch(e){}
  }
  document.getElementById('send').onclick = async function(){
    var m = document.getElementById('msg');
    if (!confirm('Send current location by SMS to ' + (recipients.join(', ') || 'recipients') + '?')) return;
    m.textContent = 'Sending...';
    try{
      var r = await fetch('/api/send_location', {method:'POST'});
      var d = await r.json();
      m.textContent = d.status === 'queued' ? 'SMS queued to ' + d.recipients.join(', ') : (d.message || d.status);
    }catch(e){ m.textContent = 'Failed to send.'; }
  };
  refresh(); setInterval(refresh, 2000);
</script></body></html>"""

# ── Routes ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/monitor")
def monitor():
    if is_system_locked():
        return redirect("/")
    return render_template("monitor.html",
                           driver=session["driver_name"],
                           plate=session["plate_number"],
                           login_time=session["login_time"])

@app.route("/report")
def report():
    if is_system_locked():
        return redirect("/")
    return render_template("report.html", session=session)

@app.route("/attendance")
def attendance():
    return render_template("attendance.html")

@app.route("/gps")
def gps_page():
    return Response(GPS_PAGE, mimetype="text/html")

# ── Kiosk Exit Route ────────────────────────────────────
@app.route("/exit_kiosk")
def exit_kiosk():
    try:
        subprocess.run(["pkill", "-f", "chromium"], check=False)
        return "Exiting kiosk mode...", 200
    except Exception as e:
        return f"Error: {e}", 500

# ── Live Camera Stream ─────────────────────────────────
def gen_frames():
    last = -1
    while True:
        seq = camera.wait_new(last, timeout=2.0)
        if seq == last:
            continue
        jseq, data = camera.get_jpeg()
        if data is None or jseq == last:
            continue
        last = jseq
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n'
               b'Content-Length: ' + str(len(data)).encode() + b'\r\n\r\n' + data + b'\r\n')
        time.sleep(0.04)

@app.route("/video_feed")
def video_feed():
    resp = Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"
    return resp

@app.route("/driver_photo/<path:filename>")
def driver_photo(filename):
    if not os.path.exists(os.path.join(DRIVER_PHOTOS, filename)):
        return "", 404
    return send_from_directory(DRIVER_PHOTOS, filename)

# ── API Endpoints ──────────────────────────────────────
@app.route("/api/start_breath_test", methods=["POST", "GET"])
def start_breath_test():
    send_to_esp32("START_TEST")
    return jsonify({"status": "success", "data": esp32_data})

@app.route("/api/breath_status")
def breath_status_route():
    connected = (time.time() - esp32_last_seen) < BT_STALE_AFTER_SEC
    info   = get_lock_info()
    locked = is_system_locked()
    return jsonify({
        "connected": connected,
        "state": "lockout" if locked else str(esp32_data.get("state", "IDLE")).lower(),
        "value": esp32_data.get("val", 0),
        "result": "locked" if locked else breath_session["result"],
        "peak": (info.get("level") or breath_session["peak"]) if locked else breath_session["peak"],
        "locked": locked,
    })

@app.route("/api/system_status")
def system_status():
    locked = is_system_locked()
    active = is_engine_active() and not locked
    state = "lockout" if locked else str(esp32_data.get("state", "IDLE")).lower()
    off_at = session["engine_off_at"]
    grace, grace_ms = 0, -1
    if off_at and state == "idle":
        left = GRACE_SECONDS - (datetime.now() - off_at).total_seconds()
        grace = max(0, math.ceil(left))
        grace_ms = max(0, int(left * 1000))
    # Cooldown only exists AFTER the admin reset (ESP32 state "COOLDOWN", val = seconds left)
    cd_state = (state == "cooldown")
    cd_left = int(_num(esp32_data.get("val", 0), float, 0) or 0) if cd_state else 0
    return jsonify({
        "state": state,
        "locked": locked,
        "lock_latched": get_persisted_lock(),
        "esp_state": str(esp32_data.get("state", "IDLE")).lower(),
        "active": active,
        "connected": (time.time() - esp32_last_seen) < BT_STALE_AFTER_SEC,
        "grace": grace,
        "grace_left_ms": grace_ms,
        "grace_total": GRACE_SECONDS,
        "engine_off": bool(off_at),
        "session_ended": bool(session["logged"]),
        "admin_reset_required": locked,
        "cooldown": cd_state,
        "cooldown_left": cd_left,
        "cooldown_total": LOCKOUT_COOLDOWN_SEC,
    })

def do_unlock(source):
    t0 = time.time()
    lock_ctl.update(source=source, source_at=t0)
    lock_ctl["unlocking_until"] = t0 + UNLOCK_WAIT_SEC + 3
    last_cmd = 0.0
    confirmed = False
    while time.time() - t0 < UNLOCK_WAIT_SEC:
        esp_state = str(esp32_data.get("state", "")).upper()
        fresh = esp32_last_seen > t0
        if fresh and esp_state != "LOCKOUT":
            confirmed = True
            break
        if time.time() - last_cmd >= 1.0:
            send_to_esp32("RESET")
            last_cmd = time.time()
        time.sleep(0.1)
    waited = int((time.time() - t0) * 1000)
    if not confirmed:
        return "timeout", "ESP32 did not confirm the reset. Check the Bluetooth link or tap the admin RFID card.", waited
    clear_persisted_lock(source)
    return "success", "System unlocked", waited

@app.route("/api/unlock_system", methods=["POST", "GET"])
def unlock_system():
    if not admin_ok():
        return jsonify({"status": "unauthorized", "message": "Admin PIN required"}), 401
    status, message, waited = do_unlock("admin")
    if status != "success":
        return jsonify({"status": status, "message": message}), 504
    return jsonify({"status": "success", "message": "System unlocked by admin", "waited_ms": waited})

@app.route("/api/lock_status")
def lock_status():
    info = get_lock_info()
    info["system_locked"] = is_system_locked()
    info["esp_state"] = str(esp32_data.get("state", "IDLE")).lower()
    return jsonify(info)

@app.route("/api/gps")
def api_gps():
    d = gps.snapshot()
    d["sms_recipients"] = [normalize_ph_number(n) for n in SMS_RECIPIENTS]
    return jsonify(d)

@app.route("/api/send_location", methods=["POST"])
def send_location():
    global last_location_sms
    now = time.time()
    if now - last_location_sms < SMS_LOCATION_COOLDOWN_SEC:
        wait = int(SMS_LOCATION_COOLDOWN_SEC - (now - last_location_sms))
        return jsonify({"status": "too_soon", "message": f"Please wait {wait}s before sending again."})
    last_location_sms = now
    gsm.send_sms(build_location_sms(session["driver_name"], session["plate_number"]), priority=0)
    return jsonify({"status": "queued",
                    "recipients": [normalize_ph_number(n) for n in SMS_RECIPIENTS],
                    "has_fix": gps.snapshot()["fix"]})

@app.route("/api/gsm_status")
def gsm_status():
    return jsonify({
        "status": gsm.status,
        "port": gsm.port,
        "sim_number": normalize_ph_number(GSM_SIM_NUMBER),
        "last_sent": gsm.last_sent,
        "last_error": gsm.last_error,
        "recipients": [normalize_ph_number(n) for n in SMS_RECIPIENTS],
        "queued": gsm.q.qsize(),
    })

@app.route("/api/gsm_diag")
def gsm_diag():
    return jsonify(gsm.diagnose())

@app.route("/api/gsm_reconnect", methods=["POST", "GET"])
def gsm_reconnect():
    with gsm.io_lock:
        ok = gsm.connect()
    return jsonify({"ok": ok, "status": gsm.status, "port": gsm.port, "last_error": gsm.last_error})

@app.route("/api/test_sms", methods=["POST", "GET"])
def test_sms():
    gsm.send_sms(build_test_sms(), priority=0)
    return jsonify({"status": "queued",
                    "sender_sim": normalize_ph_number(GSM_SIM_NUMBER),
                    "recipients": [normalize_ph_number(n) for n in SMS_RECIPIENTS]})

@app.route("/api/scan_face", methods=["POST"])
def scan_face():
    if is_system_locked():
        return jsonify({"status": "locked", "reason": "locked"})
    import face_recognition
    db = load_drivers()
    if not db["encodings"]:
        print("[scan] No drivers registered. Use 'Sync New Drivers'.")
        return jsonify({"status": "no_match", "reason": "empty_db"})
    matched_name = matched_plate = None
    best_distance, faces_seen, attempt = 9.9, 0, 0
    deadline = time.time() + SCAN_TIMEOUT_SEC
    while time.time() < deadline:
        attempt += 1
        frame = camera.read()
        if frame is None:
            time.sleep(0.1)
            continue
        try:
            scale = 0.5 if attempt % 2 else 0.75
            small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            rgb_eq = np.ascontiguousarray(cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2RGB), dtype=np.uint8)
            locs = face_recognition.face_locations(rgb_eq, model="hog")
            if not locs:
                time.sleep(0.05)
                continue
            faces_seen += 1
            top, right, bottom, left = max(locs, key=lambda l: (l[2] - l[0]) * (l[1] - l[3]))
            full_loc = [(int(top / scale), int(right / scale), int(bottom / scale), int(left / scale))]
            full_rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), dtype=np.uint8)
            enc = face_recognition.face_encodings(full_rgb, full_loc)
            if not enc:
                continue
            distances = face_recognition.face_distance(db["encodings"], enc[0])
            best_idx = int(np.argmin(distances))
            best_distance = min(best_distance, float(distances[best_idx]))
            if distances[best_idx] < FACE_MATCH_THRESHOLD:
                matched_name  = db["names"][best_idx]
                matched_plate = db["plates"][best_idx]
                break
        except Exception as e:
            print(f"[scan] error on attempt {attempt}: {e}")
            time.sleep(0.1)
            continue
        time.sleep(0.05)
    print(f"[scan] attempts={attempt} faces_seen={faces_seen} best_distance={best_distance:.3f} matched={matched_name}")
    if matched_name:
        with session_lock:
            prev_open = bool(session["driver_name"] and session["started"] and not session["logged"])
        if prev_open:
            finalize_session(session["engine_off_at"])
        login_dt = datetime.now()
        lat, lon = gps_pos()
        with session_lock:
            session.update(
                driver_name=matched_name, plate_number=matched_plate,
                login_dt=login_dt, login_time=login_dt.strftime(TIME_FMT),
                login_lat=lat, login_lon=lon,
                alerts=[], warn_events=[], yawns=0, videos=0,
                alcohol_result=None, alcohol_level="",
                current_status="AWAKE",
                started=False, engine_off_at=None, logged=False, summary=None)
        breath_session["awaiting"] = True
        breath_session["result"] = None
        breath_session["peak"] = 0
        breath_session["max"] = 0
        if SMS_NOTIFY_TIMEIN:
            gsm.send_sms(build_timein_sms(matched_name, matched_plate))
        return jsonify({"status": "matched", "name": matched_name, "plate": matched_plate})
    return jsonify({"status": "no_match", "reason": "no_face" if faces_seen == 0 else "no_match"})

@app.route("/api/sync_drivers", methods=["POST"])
def sync_drivers():
    added, skipped = sync_new_drivers()
    return jsonify({"added": added, "skipped": skipped})

@app.route("/api/detect", methods=["GET", "POST"])
def detect():
    base = {
        "alert_count":   len(session["alerts"]),
        "alerts":        list(session["alerts"]),
        "warning_count": len(session["warn_events"]),
        "yawns":         session["yawns"],
        "drowsy_limit":  DROWSY_SECONDS,
        "warn_limit":    WARN_SECONDS,
        "drowsy_for":    0.0,
        "recording":     recorder.active,
        "signs":         [],
        "detected":      [],
    }
    if is_system_locked():
        return jsonify({**base, "status": "LOCKED", "condition": "Locked"})
    if not monitoring_on():
        return jsonify({**base, "status": "INACTIVE", "condition": "Engine off"})
    snap = driver_monitor.snapshot()
    if snap["status"] in ("INACTIVE", "LOCKED") or time.time() - snap["updated"] > 5:
        return jsonify({**base, "status": "NO_DRIVER", "condition": "Starting detection..."})
    return jsonify({**base, "status": snap["status"], "condition": snap["condition"],
                    "signs": snap["signs"], "drowsy_for": snap["drowsy_for"],
                    "detected": snap["detected"]})

@app.route("/api/end_session", methods=["POST"])
def end_session():
    summary = finalize_session(session["engine_off_at"])
    send_to_esp32("DROWSY_OFF")
    if not summary:
        return jsonify({"error": "no_session"}), 400
    return jsonify(summary)

@app.route("/api/attendance")
def api_attendance():
    rows = read_csv_rows(ATTENDANCE_CSV)
    for r in rows:
        r["Status"] = r.get("Status") or "COMPLETED"
        r["Login Location"] = maps_link(r.get("Login Lat"), r.get("Login Lon"))
    rows.reverse()
    with session_lock:
        if session["driver_name"] and session["started"] and not session["logged"] and session["login_dt"]:
            rows.insert(0, {
                "Name": session["driver_name"], "Plate": session["plate_number"],
                "Date": session["login_dt"].strftime(DATE_FMT), "Login": session["login_dt"].strftime(TIME_FMT),
                "Logout": "", "Duration": "-", "Alerts": len(session["alerts"]),
                "Warnings": len(session["warn_events"]), "Rating": rating_for(len(session["alerts"])),
                "Alcohol Result": session["alcohol_result"] or "", "Alcohol Level": session["alcohol_level"],
                "Status": "ON DUTY",
                "Login Location": maps_link(session["login_lat"], session["login_lon"]),
            })
    return jsonify(rows[:1000])

@app.route("/api/events")
def api_events():
    ev = []
    for r in read_csv_rows(BREATH_LOG_CSV):
        res = r.get("Result") or ""
        if res == "PASSED":            event = "CLEAR (PASSED)"
        elif res.startswith("FAILED"): event = "ALCOHOL DETECTED"
        else:                          event = res
        ev.append({"Date": r.get("Date", ""), "Time": r.get("Time", ""), "Driver": r.get("Driver", ""),
                   "Plate": r.get("Plate", ""), "Category": "Alcohol", "Event": event,
                   "Detail": r.get("Note", ""), "Alcohol Level": r.get("Level", ""),
                   "Location": maps_link(r.get("Latitude"), r.get("Longitude"))})
    for r in read_csv_rows(DROWSY_LOG_CSV):
        event = "DROWSY ALERT" if r.get("Type") == "Drowsy Alert" else "WARNING"
        ev.append({"Date": r.get("Date", ""), "Time": r.get("Time", ""), "Driver": r.get("Driver", ""),
                   "Plate": r.get("Plate", ""), "Category": "Drowsiness", "Event": event,
                   "Detail": r.get("Detail", ""), "Alcohol Level": "",
                   "Location": maps_link(r.get("Latitude"), r.get("Longitude"))})
    def key(e):
        try:
            return datetime.strptime(f"{e['Date']} {e['Time']}", f"{DATE_FMT} {TIME_FMT}")
        except Exception:
            return datetime.min
    ev.sort(key=key, reverse=True)
    return jsonify(ev[:500])

# ── Admin Page & APIs ──────────────────────────────────
@app.route("/admin")
def admin_page():
    return send_from_directory(os.path.join(BASE_DIR, "templates"), "admin.html")

@app.route("/api/admin/overview")
@admin_required
def admin_overview():
    locked = is_system_locked()
    with session_lock:
        sess = {"driver": session["driver_name"], "plate": session["plate_number"],
                "login_time": session["login_time"], "alerts": len(session["alerts"]),
                "warnings": len(session["warn_events"]), "yawns": session["yawns"],
                "videos": session["videos"], "alcohol_result": session["alcohol_result"],
                "started": session["started"], "logged": session["logged"]}
    g = gps.snapshot()
    return jsonify({
        "locked": locked,
        "lock": get_lock_info(),
        "state": "lockout" if locked else str(esp32_data.get("state", "IDLE")).lower(),
        "esp_connected": (time.time() - esp32_last_seen) < BT_STALE_AFTER_SEC,
        "engine_active": is_engine_active() and not locked,
        "level": esp32_data.get("val", 0),
        "recording": recorder.active,
        "session": sess,
        "monitor": driver_monitor.snapshot() if monitoring_on() else {"status": "INACTIVE", "condition": ""},
        "gps": g,
        "gsm": {"status": gsm.status, "port": gsm.port, "last_sent": gsm.last_sent, "last_error": gsm.last_error},
        "recipients": [normalize_ph_number(n) for n in SMS_RECIPIENTS],
    })

@app.route("/api/admin/records")
@admin_required
def admin_records():
    kind = request.args.get("type", "breath")
    if kind not in ADMIN_LOGS:
        return jsonify({"error": "unknown type"}), 400
    path, header = ADMIN_LOGS[kind]
    rows = read_csv_rows(path)
    rows.reverse()
    total = len(rows)
    limit = request.args.get("limit", type=int)
    if limit:
        rows = rows[:limit]
    if kind == "videos":
        for r in rows:
            rel = (r.get("File") or "").split("/")
            r["Available"] = "1" if rel and os.path.isfile(os.path.join(RECORDINGS_DIR, *rel)) else "0"
    if kind == "trips":
        for r in rows:
            r["Status"] = r.get("Status") or "COMPLETED"
    return jsonify({"columns": header, "rows": rows, "total": total})

@app.route("/api/admin/export")
@admin_required
def admin_export():
    kind = request.args.get("type", "breath")
    if kind not in ADMIN_LOGS:
        return jsonify({"error": "unknown type"}), 400
    path, header = ADMIN_LOGS[kind]
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(header)
    for r in read_csv_rows(path):
        w.writerow([r.get(c, "") for c in header])
    return Response(out.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=alertx_{kind}.csv"})

@app.route("/recordings/<path:filename>")
@admin_required
def recording_file(filename):
    return send_from_directory(RECORDINGS_DIR, filename, mimetype="video/mp4",
                               as_attachment=bool(request.args.get("dl")), conditional=True)

def gps_logger():
    while True:
        time.sleep(GPS_LOG_INTERVAL_SEC)
        try:
            s = gps.snapshot()
            if s["fix"] and session["driver_name"] and not session["logged"] and is_engine_active():
                now = datetime.now()
                name, plate = _who()
                append_csv(GPS_LOG_CSV, GPS_HEADER,
                           [now.strftime(DATE_FMT), now.strftime(TIME_FMT), name, plate,
                            round(s["lat"], 6), round(s["lon"], 6), s["speed_kmh"] or 0, s["sats"]])
        except Exception as e:
            print(f"[gps-log] error: {e}")

# ── Incoming SMS Handler ("RESET") ────────────────────
def handle_incoming_sms(sender, text):
    if not re.match(r"^\W*RESET\b", (text or "").strip(), re.IGNORECASE):
        print(f"[sms-in] ignored message from {sender}")
        return
    allowed = {normalize_ph_number(n) for n in SMS_RECIPIENTS}
    if normalize_ph_number(sender) not in allowed:
        print(f"[sms-in] RESET from unauthorized number {sender} ignored")
        return
    if not is_system_locked():
        gsm.send_sms(build_no_lock_sms(), numbers=[sender], priority=0)
        return
    status, message, _ = do_unlock(f"SMS {normalize_ph_number(sender)}")
    if status == "success":
        gsm.send_sms(build_unlock_sms(), numbers=[sender], priority=0)
    else:
        gsm.send_sms(build_reset_failed_sms(), numbers=[sender], priority=0)

def sms_command_loop():
    gsm_boot_done.wait(timeout=60)
    last_connect_try = 0.0
    while True:
        time.sleep(SMS_POLL_SEC)
        try:
            if gsm.busy or not gsm.q.empty():
                continue                      # an outgoing SMS is pending: don't block the modem
            if not gsm.io_lock.acquire(timeout=2):
                continue
            msgs = []
            try:
                if (not gsm.dev or not gsm.dev.is_open) and time.time() - last_connect_try > 30:
                    last_connect_try = time.time()
                    gsm.connect()
                if gsm.dev and gsm.dev.is_open:
                    if time.time() - gsm.last_reg_ok > 10:
                        gsm.refresh_network()
                    msgs = gsm.read_incoming()
            finally:
                gsm.io_lock.release()
            for m in msgs:
                handle_incoming_sms(m["sender"], m["text"])
        except Exception as e:
            print(f"[sms-in] error: {e}")

# ── Test Routes (vibrator + speaker) ───────────────────   # <-- NEW
@app.route("/api/test_alert")
def test_alert():
    mode = request.args.get("mode", "warn")        # warn / drowsy / off
    if mode not in ("warn", "drowsy", "off"):
        return jsonify({"error": "mode must be warn, drowsy, or off"}), 400
    alert_out.set_mode(mode, force=True, hold=(15 if mode != "off" else 0))
    return jsonify({"mode": mode, "hold_sec": 15 if mode != "off" else 0,
                    "voice_ok": alert_out.voice_ok, "vib_ready": alert_out.vib is not None,
                    "audio_device": alert_out.audio_dev, "working_player": alert_out.good})

@app.route("/api/test_sound")
def test_sound():
    which = request.args.get("which", "warn")      # warn / drowsy
    path = alert_out.drowsy_wav if which == "drowsy" else alert_out.warn_wav
    alert_out._stop_sound()
    report = []
    ok = alert_out._run_chain(path, alert_out.gen, report)
    return jsonify({"ok": ok, "working_player": alert_out.good, "file": path,
                    "file_exists": os.path.isfile(path), "usb_device": alert_out.audio_dev,
                    "attempts": report, "sound_cards": aplay_list()})

# ── Start Background Threads ───────────────────────────
threading.Thread(target=read_esp32_bt, daemon=True).start()
threading.Thread(target=detector_loop, daemon=True).start()
threading.Thread(target=gps_logger, daemon=True).start()
threading.Thread(target=sms_command_loop, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)