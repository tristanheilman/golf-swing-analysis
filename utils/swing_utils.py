import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import os
import urllib.request

# Fixed golf ball position
BALL_POS = (650, 927)
COLORS = {
    "light": (255, 255, 0),
    "marks": (0, 255, 255),
    "join": (0, 20, 200),
    "axis": (200, 200, 200),
    "text": (0, 255, 0)
}

# Create PoseLandmarker once at module level
_pose_detector = None

def _ensure_model_file():
    """Download pose landmarker model if it doesn't exist."""
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "pose_landmarker_full.task")

    if not os.path.exists(model_path):
        print("Downloading pose landmarker model...")
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
        urllib.request.urlretrieve(url, model_path)
        print(f"Model downloaded to {model_path}")

    return model_path

def _get_pose_detector():
    global _pose_detector
    if _pose_detector is None:
        model_path = _ensure_model_file()
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5
        )
        _pose_detector = vision.PoseLandmarker.create_from_options(options)
    return _pose_detector

def process_frame(frame, frame_size):
    frame_w, frame_h = frame_size
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get the detector
    detector = _get_pose_detector()

    # Convert frame to MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Detect pose
    detection_result = detector.detect(mp_image)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # Draw X/Y axis in bottom-left corner
    _draw_axes(frame_bgr, origin=(100, frame_h - 100))

    if detection_result.pose_landmarks:
        landmarks = detection_result.pose_landmarks[0]

        # Extract points using landmark indices
        r_ear = _get_point_from_landmark(landmarks[8], frame_w, frame_h)  # RIGHT_EAR
        r_wrist = _get_point_from_landmark(landmarks[16], frame_w, frame_h)  # RIGHT_WRIST
        r_hip = _get_point_from_landmark(landmarks[24], frame_w, frame_h)  # RIGHT_HIP

        # Draw lines
        cv2.line(frame_bgr, r_hip, r_ear, COLORS["join"], 2, cv2.LINE_AA)
        cv2.line(frame_bgr, r_hip, r_wrist, COLORS["join"], 2, cv2.LINE_AA)
        cv2.line(frame_bgr, r_ear, r_wrist, COLORS["join"], 2, cv2.LINE_AA)
        cv2.line(frame_bgr, BALL_POS, r_wrist, COLORS["light"], 2, cv2.LINE_AA)

        # Draw points
        for point in [r_ear, r_wrist, r_hip, BALL_POS]:
            cv2.circle(frame_bgr, point, 3, COLORS["marks"], -1)

        # Calculate and display angle at hip (ear-hip-wrist)
        angle = _calculate_angle(r_ear, r_hip, r_wrist)
        cv2.putText(frame_bgr, f"Angle: {int(angle)} deg", (r_hip[0] + 20, r_hip[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["text"], 2, cv2.LINE_AA)

    return frame_bgr

def _get_point_from_landmark(landmark, frame_w, frame_h):
    x = int(landmark.x * frame_w)
    y = int(landmark.y * frame_h)
    return (x, y)

def _calculate_angle(a, b, c):
    """Calculate angle ABC (in degrees) given three points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return angle

def _draw_axes(frame, origin=(100, 100), length=50):
    """Draw X (horizontal) and Y (vertical) axis lines."""
    x_axis_end = (origin[0] + length, origin[1])
    y_axis_end = (origin[0], origin[1] - length)
    cv2.line(frame, origin, x_axis_end, COLORS["axis"], 2)  # X-axis
    cv2.line(frame, origin, y_axis_end, COLORS["axis"], 2)  # Y-axis
    cv2.putText(frame, "X", (x_axis_end[0] + 10, x_axis_end[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["axis"], 2)
    cv2.putText(frame, "Y", (y_axis_end[0] - 15, y_axis_end[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["axis"], 2)
