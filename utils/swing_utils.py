import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import os
import json
import urllib.request

COLORS = {
    "light": (255, 255, 0),
    "marks": (0, 255, 255),
    "join": (0, 20, 200),
    "axis": (200, 200, 200),
    "text": (0, 255, 0),
    "text2": (0, 200, 255),
}

# MediaPipe Pose landmark indices
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_KNEE = 25
RIGHT_KNEE = 26
RIGHT_EAR = 8

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

def detect_ball(frame, frame_size):
    """
    Attempt to detect a golf ball in the frame using Hough circle detection.
    Color-agnostic — detects by shape (small circle) in the lower portion of the frame.

    Returns (x, y) pixel tuple if found, or None if no confident detection.
    """
    frame_w, frame_h = frame_size
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Only search the lower 60% of the frame — ball is on the ground
    search_top = int(frame_h * 0.4)
    roi = blurred[search_top:, :]

    # Golf balls are small: expect radius roughly 0.5–2% of frame width
    min_radius = max(5,  int(frame_w * 0.005))
    max_radius = max(20, int(frame_w * 0.02))

    circles = cv2.HoughCircles(
        roi,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_radius * 4,
        param1=60,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        return None

    circles = np.round(circles[0]).astype(int)

    # Pick the circle with the highest Y value (lowest in frame = on the ground)
    best = max(circles, key=lambda c: c[1])
    x, y = best[0], best[1] + search_top  # offset back to full-frame coords
    return (x, y)


def process_frame(frame, frame_size, ball_pos=None, wrist_speed=None, wrist_trail=None, frame_num=0, fps=30):
    """Process a single frame. Returns (annotated_frame, metrics_dict).

    Args:
        ball_pos:     (x, y) pixel position of the golf ball, or None to skip ball annotation.
        wrist_speed:  Wrist speed in px/frame from the previous frame (for HUD display).
        wrist_trail:  List of recent (x, y) wrist positions to draw as a fading path.
        frame_num:    Current frame index (for HUD timestamp).
        fps:          Video frame rate (for HUD timestamp).
    """
    frame_w, frame_h = frame_size
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    detector = _get_pose_detector()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    detection_result = detector.detect(mp_image)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    _draw_axes(frame_bgr, origin=(100, frame_h - 100))

    # Draw wrist trail behind everything else
    if wrist_trail:
        n = len(wrist_trail)
        for i, pos in enumerate(wrist_trail):
            t = i / n  # 0 = oldest, 1 = newest
            brightness = int(60 + t * 195)
            radius = max(2, int(2 + t * 4))
            cv2.circle(frame_bgr, pos, radius, (0, brightness, brightness), -1)

    # HUD: frame counter, elapsed time, wrist speed
    _draw_hud(frame_bgr, frame_num, fps, wrist_speed)

    metrics = {}

    if detection_result.pose_landmarks:
        landmarks = detection_result.pose_landmarks[0]

        # Extract key points
        r_ear      = _get_point(landmarks[RIGHT_EAR], frame_w, frame_h)
        r_wrist    = _get_point(landmarks[RIGHT_WRIST], frame_w, frame_h)
        r_hip      = _get_point(landmarks[RIGHT_HIP], frame_w, frame_h)
        l_hip      = _get_point(landmarks[LEFT_HIP], frame_w, frame_h)
        l_shoulder = _get_point(landmarks[LEFT_SHOULDER], frame_w, frame_h)
        r_shoulder = _get_point(landmarks[RIGHT_SHOULDER], frame_w, frame_h)

        mid_hip      = ((l_hip[0] + r_hip[0]) // 2,      (l_hip[1] + r_hip[1]) // 2)
        mid_shoulder = ((l_shoulder[0] + r_shoulder[0]) // 2, (l_shoulder[1] + r_shoulder[1]) // 2)

        # --- Angle: ear-hip-wrist (existing) ---
        ear_hip_wrist = _calculate_angle(r_ear, r_hip, r_wrist)

        # --- Hip rotation: angle of hip line relative to horizontal ---
        # left_hip → mid_hip → horizontal reference point
        h_ref = (mid_hip[0] + 100, mid_hip[1])
        hip_rotation = _calculate_angle(l_hip, mid_hip, h_ref)

        # --- Shoulder tilt: angle of shoulder line relative to vertical ---
        # left_shoulder → mid_shoulder → vertical reference above mid_shoulder
        v_ref_shoulder = (mid_shoulder[0], mid_shoulder[1] - 100)
        shoulder_tilt = _calculate_angle(l_shoulder, mid_shoulder, v_ref_shoulder)

        # --- Spine angle: angle of mid_hip→mid_shoulder relative to vertical ---
        # mid_hip → mid_shoulder → vertical reference above mid_shoulder
        spine_angle = _calculate_angle(mid_hip, mid_shoulder, v_ref_shoulder)

        # Draw skeleton
        cv2.line(frame_bgr, r_hip, r_ear, COLORS["join"], 2, cv2.LINE_AA)
        cv2.line(frame_bgr, r_hip, r_wrist, COLORS["join"], 2, cv2.LINE_AA)
        cv2.line(frame_bgr, r_ear, r_wrist, COLORS["join"], 2, cv2.LINE_AA)
        if ball_pos:
            cv2.line(frame_bgr, ball_pos, r_wrist, COLORS["light"], 2, cv2.LINE_AA)
        cv2.line(frame_bgr, l_hip, r_hip, COLORS["axis"], 2, cv2.LINE_AA)          # hip line
        cv2.line(frame_bgr, mid_hip, mid_shoulder, COLORS["axis"], 2, cv2.LINE_AA) # spine line

        # Draw key points
        key_points = [r_ear, r_wrist, r_hip, l_hip, mid_hip, mid_shoulder]
        if ball_pos:
            key_points.append(ball_pos)
        for point in key_points:
            cv2.circle(frame_bgr, point, 3, COLORS["marks"], -1)

        # Annotate angles
        cv2.putText(frame_bgr, f"Ear-Hip-Wrist: {int(ear_hip_wrist)}", (r_hip[0] + 20, r_hip[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["text"], 2, cv2.LINE_AA)
        cv2.putText(frame_bgr, f"Hip Rot: {int(hip_rotation)}", (mid_hip[0] + 20, mid_hip[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["text2"], 2, cv2.LINE_AA)
        cv2.putText(frame_bgr, f"Spine: {int(spine_angle)}", (mid_shoulder[0] + 20, mid_shoulder[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["text2"], 2, cv2.LINE_AA)
        cv2.putText(frame_bgr, f"Shoulder Tilt: {int(shoulder_tilt)}", (mid_shoulder[0] + 20, mid_shoulder[1] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["text2"], 2, cv2.LINE_AA)

        metrics = {
            "wrist_x": r_wrist[0],
            "wrist_y": r_wrist[1],
            "ear_hip_wrist_deg": round(ear_hip_wrist, 1),
            "hip_rotation_deg": round(hip_rotation, 1),
            "shoulder_tilt_deg": round(shoulder_tilt, 1),
            "spine_angle_deg": round(spine_angle, 1),
        }

    return frame_bgr, metrics


def detect_swing_phases(wrist_x_list, wrist_y_list, frame_numbers=None):
    """
    Detect swing phase boundaries from wrist trajectory.
    Assumes right-handed golfer, down-the-line view.

    Args:
        wrist_x_list: wrist x pixel values per detected frame
        wrist_y_list: wrist y pixel values per detected frame
        frame_numbers: actual frame indices corresponding to each list entry.
                       Defaults to [0, 1, 2, ...].

    Returns:
        Dict mapping phase names to actual frame numbers.
    """
    n = len(wrist_x_list)
    if n < 10:
        return {}

    if frame_numbers is None:
        frame_numbers = list(range(n))

    x = np.array(wrist_x_list, dtype=float)
    y = np.array(wrist_y_list, dtype=float)

    # Aggressive smoothing to suppress setup micro-movements and per-frame jitter.
    # Use at least 7 frames, up to 1/8 of the clip length.
    window = max(7, n // 8)
    kernel = np.ones(window) / window
    x_smooth = np.convolve(x, kernel, mode='same')
    y_smooth = np.convolve(y, kernel, mode='same')

    # Guard margin: convolution edge artifacts — skip the outermost window//2 frames
    # when searching for peaks, then clamp results into valid range.
    margin = window // 2

    # --- Impact: wrist at maximum y (lowest in image), search after first quarter ---
    # Restricting the search range prevents picking up a low setup position as impact.
    impact_search_start = max(margin, n // 4)
    impact_idx = impact_search_start + int(np.argmax(y_smooth[impact_search_start:n - margin]))

    # --- Top of backswing: maximum x-displacement from the starting position ---
    # This replaces the noisy zero-crossing approach. The wrist travels furthest from
    # its address x-position at the top, regardless of which direction it swings.
    x_ref = x_smooth[margin]  # use a smoothed reference near the start, not raw frame 0
    x_displacement = np.abs(x_smooth[margin:impact_idx] - x_ref)
    if len(x_displacement) > 0:
        top_idx = margin + int(np.argmax(x_displacement))
    else:
        top_idx = impact_idx // 2
    # Enforce a minimum backswing length (at least 10% of frames)
    top_idx = max(top_idx, margin + max(1, n // 10))
    top_idx = min(top_idx, impact_idx - 1)

    # --- Address end: first sustained burst of movement toward the top ---
    # Require CONSECUTIVE frames above threshold to ignore single-frame fidgets.
    dx = np.diff(x_smooth)
    dy = np.diff(y_smooth)
    speed = np.sqrt(dx**2 + dy**2)
    pre_top_speed = speed[:top_idx] if top_idx > 0 else speed
    speed_threshold = np.percentile(pre_top_speed, 60) * 1.5
    consecutive_needed = 3  # must stay above threshold for 3 frames to count
    address_end_idx = 0
    consecutive = 0
    for i in range(top_idx):
        if speed[i] > speed_threshold:
            consecutive += 1
            if consecutive >= consecutive_needed:
                address_end_idx = max(0, i - consecutive_needed)
                break
        else:
            consecutive = 0

    # --- Downswing start: when wrist x crosses back through the address plane ---
    # "The plane" = the wrist's x-position at address (x_ref).
    # Downswing (and follow-through) cannot be declared until the wrist has
    # physically returned to that x-position on the way back to impact.
    # backswing_dir tells us which side of x_ref the wrist traveled to at the top,
    # so we can detect the crossing regardless of camera orientation or handedness.
    backswing_dir = np.sign(x_smooth[top_idx] - x_ref)  # +1 = right, -1 = left
    downswing_start_idx = min(top_idx + 1, impact_idx - 1)  # safe fallback

    for i in range(top_idx, min(impact_idx + 1, n)):
        # Displaced is positive while on the backswing side of the plane,
        # zero or negative once the wrist has crossed back through.
        displaced = (x_smooth[i] - x_ref) * backswing_dir
        if displaced <= 0:
            downswing_start_idx = i
            break

    def fn(idx):
        """Map list index to actual frame number."""
        idx = max(0, min(idx, n - 1))
        return frame_numbers[idx]

    phases = {
        "address":        {"start_frame": fn(0),                    "end_frame": fn(address_end_idx)},
        "backswing":      {"start_frame": fn(address_end_idx + 1),   "end_frame": fn(top_idx - 1)},
        "top":            {"frame": fn(top_idx)},
        "downswing":      {"start_frame": fn(downswing_start_idx),   "end_frame": fn(max(downswing_start_idx, impact_idx - 1))},
        "impact":         {"frame": fn(impact_idx)},
        "follow_through": {"start_frame": fn(impact_idx + 1),        "end_frame": fn(n - 1)},
    }

    return phases


def export_swing_data(output_path, video_path, fps, phases, frame_data):
    """Export all per-frame metrics and phase summaries to JSON."""
    metrics_at_top = {}
    metrics_at_impact = {}

    # Build a lookup from frame number → frame_data entry for key-frame extraction
    frame_lookup = {fd["frame"]: fd for fd in frame_data}

    if phases:
        top_frame = phases.get("top", {}).get("frame")
        impact_frame = phases.get("impact", {}).get("frame")

        if top_frame is not None and top_frame in frame_lookup:
            fd = frame_lookup[top_frame]
            metrics_at_top = {
                "hip_rotation_deg": fd.get("hip_rotation_deg"),
                "shoulder_tilt_deg": fd.get("shoulder_tilt_deg"),
                "spine_angle_deg": fd.get("spine_angle_deg"),
            }

        if impact_frame is not None and impact_frame in frame_lookup:
            fd = frame_lookup[impact_frame]
            # Average wrist speed over a ±3 frame window around impact to reduce noise
            speed_window = 3
            speeds = []
            for i in range(impact_frame - speed_window, impact_frame + speed_window + 1):
                curr = frame_lookup.get(i)
                prev = frame_lookup.get(i - 1)
                if curr and prev:
                    dx = curr["wrist_x"] - prev["wrist_x"]
                    dy = curr["wrist_y"] - prev["wrist_y"]
                    speeds.append(math.sqrt(dx**2 + dy**2))
            wrist_speed = round(max(speeds), 2) if speeds else None
            metrics_at_impact = {
                "hip_rotation_deg": fd.get("hip_rotation_deg"),
                "wrist_speed_px_per_frame": wrist_speed,
            }

    # Tempo ratio: backswing frames / downswing frames (pro average ~3:1)
    tempo_ratio = None
    if phases:
        bs = phases.get("backswing", {})
        ds = phases.get("downswing", {})
        bs_frames = bs.get("end_frame", 0) - bs.get("start_frame", 0)
        ds_frames = ds.get("end_frame", 0) - ds.get("start_frame", 0)
        if ds_frames > 0:
            tempo_ratio = round(bs_frames / ds_frames, 2)

    swing_data = {
        "video": video_path,
        "fps": fps,
        "total_frames": len(frame_data),
        "phases": phases,
        "tempo_ratio": tempo_ratio,
        "metrics_at_top": metrics_at_top,
        "metrics_at_impact": metrics_at_impact,
        "frame_data": frame_data,
    }

    with open(output_path, "w") as f:
        json.dump(swing_data, f, indent=2)

    print(f"Swing data exported to: {output_path}")


def stamp_phase_labels(video_path, phases, fps):
    """
    Post-processing pass: read the already-written annotated video and stamp
    the current swing phase label on each frame, then replace the original file.
    """
    if not phases:
        return

    # BGR colors per phase
    PHASE_COLORS = {
        "ADDRESS":        (200, 200, 200),
        "BACKSWING":      (  0, 200, 255),
        "TOP":            (255, 255,   0),
        "DOWNSWING":      (  0, 140, 255),
        "IMPACT":         (  0,   0, 255),
        "FOLLOW_THROUGH": (  0, 200,   0),
    }

    def phase_for_frame(n):
        return ""

    cap = cv2.VideoCapture(video_path)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)

    tmp_path = video_path + ".tmp.mp4"
    writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*'mp4v'), fps_in, (w, h))

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        label = phase_for_frame(frame_num)
        if label:
            color = PHASE_COLORS.get(label, (255, 255, 255))
            font = cv2.FONT_HERSHEY_DUPLEX
            scale = 1.2
            thickness = 2
            (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
            tx = (w - tw) // 2
            ty = 58
            # Drop shadow for readability over any background
            cv2.putText(frame, label, (tx + 2, ty + 2), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
            cv2.putText(frame, label, (tx, ty),         font, scale, color,     thickness,     cv2.LINE_AA)

        writer.write(frame)
        frame_num += 1

    cap.release()
    writer.release()
    os.replace(tmp_path, video_path)
    print(f"  Phase labels stamped onto: {video_path}")


def _draw_hud(frame, frame_num, fps, wrist_speed):
    """Semi-transparent HUD panel in the top-left corner."""
    speed_str = f"{wrist_speed:.1f} px/f" if wrist_speed is not None else "--"
    lines = [
        f"Frame {frame_num}  |  {frame_num / max(fps, 1):.1f}s",
        f"Wrist Spd: {speed_str}",
    ]
    pad, line_h, box_w = 8, 22, 210
    box_h = pad * 2 + len(lines) * line_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, line in enumerate(lines):
        y = 10 + pad + (i + 1) * line_h
        cv2.putText(frame, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2.LINE_AA)


def _get_point(landmark, frame_w, frame_h):
    return (int(landmark.x * frame_w), int(landmark.y * frame_h))

# Keep old name for compatibility
def _get_point_from_landmark(landmark, frame_w, frame_h):
    return _get_point(landmark, frame_w, frame_h)

def _calculate_angle(a, b, c):
    """Calculate angle ABC (in degrees) given three points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
    cos_angle = np.dot(ba, bc) / denom
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

def _draw_axes(frame, origin=(100, 100), length=50):
    """Draw X (horizontal) and Y (vertical) axis lines."""
    x_axis_end = (origin[0] + length, origin[1])
    y_axis_end = (origin[0], origin[1] - length)
    cv2.line(frame, origin, x_axis_end, COLORS["axis"], 2)
    cv2.line(frame, origin, y_axis_end, COLORS["axis"], 2)
    cv2.putText(frame, "X", (x_axis_end[0] + 10, x_axis_end[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["axis"], 2)
    cv2.putText(frame, "Y", (y_axis_end[0] - 15, y_axis_end[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["axis"], 2)
