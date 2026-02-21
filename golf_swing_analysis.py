import cv2
import os
import math
from collections import deque
from utils.video_utils import create_video_capture, create_video_writer
from utils.swing_utils import process_frame, detect_ball, detect_swing_phases, export_swing_data, stamp_phase_labels

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv"}

input_dir = "input"
output_dir = "output"
os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

input_videos = [
    os.path.join(input_dir, f)
    for f in sorted(os.listdir(input_dir))
    if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS
]

if not input_videos:
    print(f"No video files found in '{input_dir}/'. Supported formats: {', '.join(VIDEO_EXTENSIONS)}")
    exit(0)

print(f"Found {len(input_videos)} video(s) to process: {[os.path.basename(v) for v in input_videos]}\n")


def process_video(input_video):
    base_name = os.path.splitext(os.path.basename(input_video))[0]
    output_video_path = os.path.join(output_dir, base_name + "_out_landmarks.mp4")
    output_json_path  = os.path.join(output_dir, base_name + "_swing_data.json")

    print(f"--- Processing: {input_video} ---")

    cap, fps, frame_size = create_video_capture(input_video)
    writer = create_video_writer(output_video_path, fps, frame_size)

    # Detect ball position from the first frame, once per video
    ret0, first_frame = cap.read()
    if not ret0:
        print(f"  Could not read first frame, skipping.")
        cap.release()
        writer.release()
        return
    ball_pos = detect_ball(first_frame, frame_size)
    if ball_pos:
        print(f"  Ball detected at {ball_pos}")
    else:
        print(f"  Ball not detected — ball annotation will be skipped.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind to start

    frame_data = []
    frame_num = 0
    wrist_trail = deque(maxlen=20)  # last 20 wrist positions for path trail
    prev_wrist = None               # wrist position from the previous frame
    wrist_speed_display = None      # speed computed from prev frame (shown on current frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame, metrics = process_frame(
            frame, frame_size,
            ball_pos=ball_pos,
            wrist_speed=wrist_speed_display,
            wrist_trail=list(wrist_trail),
            frame_num=frame_num,
            fps=fps,
        )
        writer.write(annotated_frame)

        if metrics:
            curr_wrist = (metrics["wrist_x"], metrics["wrist_y"])
            if prev_wrist:
                dx = curr_wrist[0] - prev_wrist[0]
                dy = curr_wrist[1] - prev_wrist[1]
                wrist_speed_display = round(math.sqrt(dx**2 + dy**2), 1)
            wrist_trail.append(curr_wrist)
            prev_wrist = curr_wrist
            frame_data.append({"frame": frame_num, **metrics})

        frame_num += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"  Processed {frame_num} frames ({len(frame_data)} with detected pose).")

    wrist_x_list  = [fd["wrist_x"] for fd in frame_data]
    wrist_y_list  = [fd["wrist_y"] for fd in frame_data]
    frame_numbers = [fd["frame"]   for fd in frame_data]

    phases = detect_swing_phases(wrist_x_list, wrist_y_list, frame_numbers)

    if phases:
        print("  Detected swing phases:")
        for phase, bounds in phases.items():
            print(f"    {phase}: {bounds}")
    else:
        print("  Could not detect swing phases (too few frames with pose data).")

    export_swing_data(output_json_path, input_video, fps, phases, frame_data)

    # Second pass: stamp current swing phase label onto each frame
    stamp_phase_labels(output_video_path, phases, fps)
    print(f"  Annotated video: {output_video_path}\n")


for video_path in input_videos:
    process_video(video_path)
