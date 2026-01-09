# main.py
"""
Main pipeline for real-time lip-sync and speech-to-text demo.
"""
import sys
import subprocess
import importlib

def install_and_import(packages):
    import importlib
    import sys
    import subprocess
    for pkg in packages:
        try:
            importlib.import_module(pkg[0])
        except ImportError:
            print(f"Installing {pkg[1]}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg[1]])

# List of (import_name, pip_name)
REQUIRED_PACKAGES = [
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("transformers", "transformers"),
    ("sounddevice", "sounddevice"),
    ("scipy", "scipy"),
]

install_and_import(REQUIRED_PACKAGES)

import cv2
import numpy as np
import time
from audio import AudioStream, SpeechRecognizer
from vision import FaceMouthDetector
from lipsync import LipSyncAnalyzer

# --- Settings ---
DARK_BG = (20, 20, 20)
TEXT_COLOR = (220, 220, 220)
MOUTH_COLOR = (80, 180, 255)

# --- Init ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Webcam not found.")
    sys.exit(1)

face_mouth = FaceMouthDetector()
lipsync = LipSyncAnalyzer()
audio_stream = AudioStream()
speech_rec = SpeechRecognizer()
audio_stream.start()



import threading
spoken_text = ""
last_displayed_text = ""
text_lock = threading.Lock()
text_timer = 0
last_mouth_open = 0

# Background speech recognition thread
def speech_thread(audio_stream, speech_rec, update_interval=1.0):
    global spoken_text, last_displayed_text, text_lock
    while True:
        audio = audio_stream.read(seconds=2.5)
        text = speech_rec.transcribe(audio)
        # Only update if text is not empty and different from last
        if text and text != last_displayed_text:
            with text_lock:
                spoken_text = text
                last_displayed_text = text
        time.sleep(update_interval)

speech_bg_thread = threading.Thread(target=speech_thread, args=(audio_stream, speech_rec), daemon=True)
speech_bg_thread.start()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
        overlay = frame.copy()
        # Face & mouth detection
        face_box, mouth_box = face_mouth.detect(frame)
        # Lip sync analysis
        height_change, motion_score = lipsync.analyze(frame, mouth_box)
        mouth_open = (height_change > 2 or motion_score > 8)
        # Draw mouth region
        if mouth_box:
            x, y, w, h = mouth_box
            cv2.rectangle(overlay, (x, y), (x+w, y+h), MOUTH_COLOR, 2)
        # Blend overlay for dark bg
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        # Draw text
        if mouth_open:
            last_mouth_open = time.time()
        # Only update text if mouth is open recently
        if time.time() - last_mouth_open < 1.0:
            with text_lock:
                display_text = spoken_text
        else:
            display_text = last_displayed_text
        # Render text
        if display_text:
            (tw, th), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
            tx = (frame.shape[1] - tw) // 2
            ty = frame.shape[0] - 40
            cv2.rectangle(frame, (tx-16, ty-th-16), (tx+tw+16, ty+16), DARK_BG, -1)
            cv2.putText(frame, display_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.1, TEXT_COLOR, 2, cv2.LINE_AA)
        cv2.imshow("Lip-Sync Speech Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    audio_stream.stop()
    cap.release()
    cv2.destroyAllWindows()
