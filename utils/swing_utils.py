import cv2
import numpy as np
import mediapipe as mp
import math

mp_pose = mp.solutions.pose

# Fixed golf ball position
BALL_POS = (650, 927)
COLORS = {
    "light": (255, 255, 0),
    "marks": (0, 255, 255),
    "join": (0, 20, 200),
    "axis": (200, 200, 200),
    "text": (0, 255, 0)
}

def process_frame(frame, frame_size):
    frame_w, frame_h = frame_size
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        results = pose.process(frame_rgb)
        landmarks = results.pose_landmarks
        enum_pose = mp_pose.PoseLandmark
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Draw X/Y axis in bottom-left corner
        _draw_axes(frame_bgr, origin=(100, frame_h - 100))

        if landmarks:
            # Extract points
            r_ear = _get_point(landmarks, enum_pose.RIGHT_EAR, frame_w, frame_h)
            r_wrist = _get_point(landmarks, enum_pose.RIGHT_WRIST, frame_w, frame_h)
            r_hip = _get_point(landmarks, enum_pose.RIGHT_HIP, frame_w, frame_h)

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

def _get_point(landmarks, landmark, frame_w, frame_h):
    x = int(landmarks.landmark[landmark].x * frame_w)
    y = int(landmarks.landmark[landmark].y * frame_h)
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
