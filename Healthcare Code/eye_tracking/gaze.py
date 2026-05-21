"""
Iris-based gaze estimation using MediaPipe FaceLandmarker (Tasks API, v0.10+).

Output:
  gaze_x : float in [-1, +1]   negative = looking left, positive = right
  gaze_y : float in [-1, +1]   negative = looking up,   positive = down
"""

import os
import time
import numpy as np
import cv2

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Landmark indices (478-point Face Mesh schema)
LEFT_EYE_INNER   = 133
LEFT_EYE_OUTER   = 33
LEFT_EYE_TOP     = 159
LEFT_EYE_BOTTOM  = 145
RIGHT_EYE_INNER  = 362
RIGHT_EYE_OUTER  = 263
RIGHT_EYE_TOP    = 386
RIGHT_EYE_BOTTOM = 374
LEFT_IRIS_CENTER  = 468
RIGHT_IRIS_CENTER = 473

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "face_landmarker.task")


class GazeEstimator:
    """
    Estimates normalised gaze direction from a webcam frame.

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

        h, w = bgr_frame.shape[:2]

        def pt(idx):
            return np.array([lm[idx].x * w, lm[idx].y * h])

        gx_l, gy_l = self._eye_gaze(pt, "left")
        gx_r, gy_r = self._eye_gaze(pt, "right")
        return float((gx_l + gx_r) / 2.0), float((gy_l + gy_r) / 2.0)

    def _eye_gaze(self, pt, side: str):
        if side == "left":
            inner, outer = LEFT_EYE_INNER, LEFT_EYE_OUTER
            top, bottom  = LEFT_EYE_TOP,   LEFT_EYE_BOTTOM
            iris         = LEFT_IRIS_CENTER
        else:
            inner, outer = RIGHT_EYE_INNER, RIGHT_EYE_OUTER
            top, bottom  = RIGHT_EYE_TOP,   RIGHT_EYE_BOTTOM
            iris         = RIGHT_IRIS_CENTER

        eye_center = (pt(inner) + pt(outer)) / 2.0
        eye_width  = np.linalg.norm(pt(outer) - pt(inner))
        eye_height = np.linalg.norm(pt(bottom) - pt(top))

        if eye_width < 1 or eye_height < 1:
            return 0.0, 0.0

        iris_pt = pt(iris)
        gx = (iris_pt[0] - eye_center[0]) / (eye_width  / 2.0)
        gy = (iris_pt[1] - eye_center[1]) / (eye_height / 2.0)
        return gx, gy

    def draw_debug(self, frame: np.ndarray) -> np.ndarray:
        if self._landmarks is None:
            return frame
        h, w = frame.shape[:2]
        lm = self._landmarks
        for idx in (LEFT_EYE_INNER, LEFT_EYE_OUTER, LEFT_EYE_TOP, LEFT_EYE_BOTTOM,
                    RIGHT_EYE_INNER, RIGHT_EYE_OUTER, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM):
            cv2.circle(frame, (int(lm[idx].x * w), int(lm[idx].y * h)), 3, (255, 80, 0), -1)
        for idx in (LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER):
            cv2.circle(frame, (int(lm[idx].x * w), int(lm[idx].y * h)), 5, (0, 255, 0), -1)
        return frame

    def close(self):
        self._landmarker.close()
