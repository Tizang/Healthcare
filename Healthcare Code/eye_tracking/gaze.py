"""
Iris-based gaze estimation using MediaPipe FaceLandmarker (Tasks API, v0.10+).

Algorithm (inspired by soumyagautam/Eye-Mouse-Tracking):
  Use the absolute iris landmark position within the camera frame as gaze proxy.
  When you look left, the iris moves to the left side of the frame.
  When you look to the top-right corner, the iris moves to the top-right.

  iris_x, iris_y ∈ [0, 1]  (normalised image coordinates from MediaPipe)
  gaze_x, gaze_y ∈ [~-0.5, ~+0.5]  (centred on 0.5, raw — calibration maps to ±1)

  Both iris centres (left=468, right=473) are averaged for stability.
  Additionally, all 5 landmarks per iris ring are averaged to suppress jitter.

Output after calibration:
  gaze_x : float in [-1, +1]   negative = looking left, positive = right
  gaze_y : float in [-1, +1]   negative = looking up,   positive = down
"""

import os
import time
import numpy as np
import cv2

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Iris landmark indices in the 478-point Face Mesh
# 468–472: left iris  (center, top, right, bottom, left)
# 473–477: right iris (center, top, right, bottom, left)
LEFT_IRIS  = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

# Eye-corner landmarks — used only for draw_debug
LEFT_EYE_CORNERS  = [33, 133, 159, 145]
RIGHT_EYE_CORNERS = [263, 362, 386, 374]

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "face_landmarker.task")


class GazeEstimator:
    """
    Estimates gaze from iris position in the camera frame.

    Usage:
        est = GazeEstimator()
        gaze_x, gaze_y = est.estimate(bgr_frame)
        # returns (None, None) when no face detected
    """

    def __init__(self, min_detection_confidence: float = 0.6):
        model_path = os.path.abspath(_MODEL_PATH)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                "Download with:\n"
                "  curl -L -o face_landmarker.task "
                "https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            )

        import mediapipe as mp
        base_opts = python.BaseOptions(model_asset_path=model_path)
        opts = vision.FaceLandmarkerOptions(
            base_options=base_opts,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(opts)
        self._mp = mp
        self._landmarks = None
        self._transform_matrix = None
        self._start_time = time.time()

    def estimate(self, bgr_frame: np.ndarray):
        """Returns (gaze_x, gaze_y) or (None, None) if no face detected."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=rgb
        )
        timestamp_ms = int((time.time() - self._start_time) * 1000)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            self._landmarks = None
            self._transform_matrix = None
            return None, None

        lm = result.face_landmarks[0]
        self._landmarks = lm
        self._transform_matrix = (
            result.facial_transformation_matrixes[0]
            if result.facial_transformation_matrixes else None
        )

        # Average all 5 landmarks per iris ring for a stable centre estimate
        left_x  = np.mean([lm[i].x for i in LEFT_IRIS])
        left_y  = np.mean([lm[i].y for i in LEFT_IRIS])
        right_x = np.mean([lm[i].x for i in RIGHT_IRIS])
        right_y = np.mean([lm[i].y for i in RIGHT_IRIS])

        # Average both eyes
        iris_x = (left_x + right_x) / 2.0
        iris_y = (left_y + right_y) / 2.0

        # Map [0, 1] → [−1, +1] centred at 0.5
        # (Calibration will correct offset + scale + non-linearity)
        gaze_x = (iris_x - 0.5) * 2.0
        gaze_y = (iris_y - 0.5) * 2.0

        return float(gaze_x), float(gaze_y)

    def draw_debug(self, frame: np.ndarray) -> np.ndarray:
        if self._landmarks is None:
            return frame
        h, w = frame.shape[:2]
        lm = self._landmarks
        for idx in LEFT_EYE_CORNERS + RIGHT_EYE_CORNERS:
            cv2.circle(frame, (int(lm[idx].x * w), int(lm[idx].y * h)), 3, (255, 80, 0), -1)
        for idx in LEFT_IRIS + RIGHT_IRIS:
            cv2.circle(frame, (int(lm[idx].x * w), int(lm[idx].y * h)), 3, (0, 255, 0), -1)
        return frame

    def close(self):
        self._landmarker.close()
