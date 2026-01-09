# lipsync.py
"""
Analyzes mouth motion for lip-sync using bounding box height and frame differencing.
"""
import numpy as np
import cv2

class LipSyncAnalyzer:
    def __init__(self):
        self.prev_mouth_img = None
        self.prev_height = None
        self.motion_score = 0

    def analyze(self, frame, mouth_box):
        if mouth_box is None:
            self.prev_mouth_img = None
            self.prev_height = None
            self.motion_score = 0
            return 0, 0  # (height_change, motion_score)
        x, y, w, h = mouth_box
        mouth_img = frame[y:y+h, x:x+w]
        height_change = 0
        motion = 0
        if self.prev_height is not None:
            height_change = h - self.prev_height
        if self.prev_mouth_img is not None and mouth_img.shape == self.prev_mouth_img.shape:
            diff = cv2.absdiff(cv2.cvtColor(mouth_img, cv2.COLOR_BGR2GRAY), cv2.cvtColor(self.prev_mouth_img, cv2.COLOR_BGR2GRAY))
            motion = np.mean(diff)
        self.prev_mouth_img = mouth_img.copy()
        self.prev_height = h
        self.motion_score = 0.7 * self.motion_score + 0.3 * motion  # smooth
        return height_change, self.motion_score
