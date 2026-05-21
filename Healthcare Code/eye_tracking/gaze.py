"""
Iris-based gaze estimation using MediaPipe FaceLandmarker (Tasks API, v0.10+).

Key improvements over basic 2D iris tracking:
  - 3D gaze vector: uses depth component (lm.z) for full 3D iris offset
  - Head-pose compensation: de-rotates gaze vector into face-local space so
    head movement no longer displaces the cursor
  - 5-point iris average: uses all 5 iris landmarks per eye (not just center)
    for a more stable iris center estimate

Output:
  gaze_x : float in approx [-0.5, +0.5]   negative = looking left
  gaze_y : float in approx [-0.5, +0.5]   negative = looking up
  (calibration maps this to the full [-1, +1] control range)
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

# 5 iris landmarks per eye: center + 4 cardinal points
LEFT_IRIS  = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

# Outer eye corners — used to estimate face scale (width) for z depth
FACE_LEFT_PT  = 33
FACE_RIGHT_PT = 263

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

        # Face width in pixels — used to scale the z (depth) component so
        # all three axes are in comparable pixel units
        face_px = abs(lm[FACE_LEFT_PT].x - lm[FACE_RIGHT_PT].x) * w
        face_scale = max(face_px, 1.0)

        # Inverse rotation matrix for head-pose compensation
        R_inv = self._rotation_inv()

        gx_l, gy_l = self._eye_gaze(lm, "left",  R_inv, h, w, face_scale)
        gx_r, gy_r = self._eye_gaze(lm, "right", R_inv, h, w, face_scale)
        return float((gx_l + gx_r) / 2.0), float((gy_l + gy_r) / 2.0)

    def _rotation_inv(self):
        """
        Returns R^-1 from the facial transformation matrix.
        Multiplying a camera-space vector by R^-1 rotates it into face-local
        space, making it independent of head orientation.
        """
        if self._transform_matrix is None:
            return None
        T = np.array(self._transform_matrix)
        if T.shape != (4, 4):
            return None
        R_raw = T[:3, :3]
        # Normalise columns to remove any scale the matrix might carry
        col_norms = np.linalg.norm(R_raw, axis=0)
        col_norms = np.where(col_norms < 1e-9, 1.0, col_norms)
        R = R_raw / col_norms[np.newaxis, :]
        return R.T  # For orthogonal R: R^-1 = R^T

    def _eye_gaze(self, lm, side: str, R_inv, h: int, w: int, face_scale: float):
        if side == "left":
            inner, outer = LEFT_EYE_INNER, LEFT_EYE_OUTER
            top, bottom  = LEFT_EYE_TOP,   LEFT_EYE_BOTTOM
            iris_indices = LEFT_IRIS
        else:
            inner, outer = RIGHT_EYE_INNER, RIGHT_EYE_OUTER
            top, bottom  = RIGHT_EYE_TOP,   RIGHT_EYE_BOTTOM
            iris_indices = RIGHT_IRIS

        def pt3(idx):
            # 3D point in approximate pixel/camera space
            # z is scaled to the same order of magnitude as x, y
            return np.array([
                lm[idx].x * w,
                lm[idx].y * h,
                lm[idx].z * face_scale,
            ])

        # Iris center = average of all 5 iris landmarks
        iris_pts = np.array([pt3(i) for i in iris_indices])
        iris_pt  = iris_pts.mean(axis=0)

        inner_pt = pt3(inner)
        outer_pt = pt3(outer)
        top_pt   = pt3(top)
        bot_pt   = pt3(bottom)

        eye_center = (inner_pt + outer_pt) / 2.0
        eye_w = np.linalg.norm(outer_pt - inner_pt)
        eye_h = np.linalg.norm(bot_pt   - top_pt)

        if eye_w < 1 or eye_h < 1:
            return 0.0, 0.0

        # 3D gaze vector: direction from eye centre to iris in camera space
        gaze_vec = iris_pt - eye_center

        # Head-pose compensation: rotate into face-local frame
        # After this, head rotation no longer affects the gaze signal
        if R_inv is not None:
            gaze_vec = R_inv @ gaze_vec

        # Normalise by eye dimensions → approx [-0.5, +0.5] range
        gx = gaze_vec[0] / (eye_w / 2.0)
        gy = gaze_vec[1] / (eye_h / 2.0)
        return float(gx), float(gy)

    def draw_debug(self, frame: np.ndarray) -> np.ndarray:
        if self._landmarks is None:
            return frame
        h, w = frame.shape[:2]
        lm = self._landmarks
        for idx in (LEFT_EYE_INNER, LEFT_EYE_OUTER, LEFT_EYE_TOP, LEFT_EYE_BOTTOM,
                    RIGHT_EYE_INNER, RIGHT_EYE_OUTER, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM):
            cv2.circle(frame, (int(lm[idx].x * w), int(lm[idx].y * h)), 3, (255, 80, 0), -1)
        for idx in LEFT_IRIS + RIGHT_IRIS:
            cv2.circle(frame, (int(lm[idx].x * w), int(lm[idx].y * h)), 3, (0, 255, 0), -1)
        return frame

    def close(self):
        self._landmarker.close()
