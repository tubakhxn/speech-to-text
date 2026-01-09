# vision.py
"""
Handles face and mouth detection using OpenCV DNN or Haar Cascade (no MediaPipe).
"""
import cv2
import numpy as np

class FaceMouthDetector:
    def __init__(self, face_model_path=None, mouth_cascade_path=None):
        # Use OpenCV's default Haar cascades for face and mouth (smile as proxy for mouth)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        face_box = None
        mouth_box = None
        for (x, y, w, h) in faces:
            face_box = (x, y, w, h)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            # Use smile as proxy for mouth
            mouths = self.mouth_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.7,
                minNeighbors=22,
                minSize=(25, 15)
            )
            for (mx, my, mw, mh) in mouths:
                # Only consider mouths in lower half of face
                if my > h//2:
                    mouth_box = (x+mx, y+my, mw, mh)
                    break
            break  # Only first face
        return face_box, mouth_box
