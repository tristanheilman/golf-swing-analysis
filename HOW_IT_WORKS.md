# How It Works

A walkthrough of the golf swing analysis pipeline — from raw video in to annotated video and structured metrics out.

---

## Project Structure

```
golf-swing-analysis/
├── golf_swing_analysis.py       # Entry point — orchestrates the full pipeline
├── utils/
│   ├── video_utils.py           # Video I/O helpers (open/read/write)
│   └── swing_utils.py           # All pose detection, angle math, phase detection, export
├── input/
│   └── Reference_Swing_DTL.mp4  # Source video (down-the-line view)
├── output/
│   ├── *_out_landmarks.mp4      # Annotated video (generated)
│   └── *_swing_data.json        # Per-frame metrics + phase data (generated)
├── models/
│   └── pose_landmarker_full.task # MediaPipe model (auto-downloaded on first run)
└── requirements.txt
```

---

## End-to-End Flow

```
input/*.mp4
    │
    ▼
[1] video_utils.py — open video, read fps + frame dimensions
    │
    ▼
[2] swing_utils.py — for each frame:
    │   ├── MediaPipe PoseLandmarker detects 33 body landmarks
    │   ├── Extract key pixel coordinates (hip, shoulder, wrist, ear)
    │   ├── Compute 4 angles per frame
    │   ├── Draw skeleton lines + angle labels onto frame
    │   └── Return annotated frame + metrics dict
    │
    ▼
[3] golf_swing_analysis.py — collect per-frame metrics into frame_data[]
    │
    ▼
[4] swing_utils.detect_swing_phases() — post-process wrist trajectory
    │   ├── Smooth wrist x/y with moving average
    │   ├── Find impact frame  (max wrist y = lowest point in image)
    │   ├── Find top-of-backswing (last x-direction reversal before impact)
    │   └── Find address end  (first frame where wrist speed exceeds threshold)
    │
    ▼
[5] swing_utils.export_swing_data() — write JSON
    │   ├── Phase boundaries (frame numbers)
    │   ├── Key-frame snapshots (metrics at top + impact)
    │   └── Full frame_data[] array
    │
    ▼
output/*_out_landmarks.mp4   (annotated video)
output/*_swing_data.json     (structured metrics)
```

---

## Working Pieces

### 1. `golf_swing_analysis.py` — Entry Point

The script that runs everything. It:

1. Opens the input video via `create_video_capture()` and creates an output writer via `create_video_writer()`.
2. Loops frame-by-frame, calling `process_frame()` for each one.
3. Writes the annotated frame to the output video.
4. Accumulates a `frame_data` list of per-frame metric dicts (only for frames where a pose was detected).
5. After the loop, extracts `wrist_x_list`, `wrist_y_list`, and `frame_numbers` from `frame_data` and passes them to `detect_swing_phases()`.
6. Calls `export_swing_data()` to write the final JSON.

---

### 2. `utils/video_utils.py` — Video I/O

Two thin wrappers around OpenCV:

| Function | What it does |
|---|---|
| `create_video_capture(path)` | Opens an MP4, reads fps + dimensions, returns `(cap, fps, frame_size)` |
| `create_video_writer(path, fps, frame_size)` | Creates an MP4 writer with the `mp4v` codec |

---

### 3. `utils/swing_utils.py` — Core Logic

This is where all the real work happens.

#### Model Loading — `_get_pose_detector()`

- Loads the MediaPipe `PoseLandmarker` model once and caches it in a module-level global.
- On first run, `_ensure_model_file()` auto-downloads `pose_landmarker_full.task` into `models/` if it isn't there.
- The model runs in image (non-streaming) mode with `min_pose_detection_confidence=0.5`.

#### Per-Frame Processing — `process_frame(frame, frame_size)`

Returns `(annotated_frame, metrics_dict)`. For each frame:

1. **Color conversion** — OpenCV reads BGR; MediaPipe expects RGB. Converted in and back out.
2. **Pose detection** — The model returns 33 normalized landmarks `(x, y, z, visibility)`. `x` and `y` are in `[0.0, 1.0]`; multiplied by frame width/height to get pixel coordinates.
3. **Key points extracted** (by landmark index):

   | Landmark | Index | Used for |
   |---|---|---|
   | RIGHT_EAR | 8 | Ear-hip-wrist angle |
   | LEFT_SHOULDER | 11 | Shoulder tilt, mid-shoulder |
   | RIGHT_SHOULDER | 12 | Mid-shoulder |
   | LEFT_HIP | 23 | Hip rotation, mid-hip |
   | RIGHT_HIP | 24 | Ear-hip-wrist, mid-hip |
   | RIGHT_WRIST | 16 | All wrist tracking + ear-hip-wrist |

   Two derived points are computed:
   - `mid_hip` = midpoint of left and right hip
   - `mid_shoulder` = midpoint of left and right shoulder

4. **Angles computed** via `_calculate_angle(a, b, c)` — returns angle at vertex `b` between rays `b→a` and `b→c` using the dot-product formula:

   | Metric | Triplet | Reference | What it reveals |
   |---|---|---|---|
   | **Ear-Hip-Wrist** | `r_ear → r_hip → r_wrist` | — | Overall arm/body connection angle |
   | **Hip Rotation** | `l_hip → mid_hip → h_ref` | Horizontal point at `(mid_hip.x + 100, mid_hip.y)` | How level/tilted the hips are |
   | **Shoulder Tilt** | `l_shoulder → mid_shoulder → v_ref` | Vertical point at `(mid_shoulder.x, mid_shoulder.y - 100)` | How much the shoulders are tilted |
   | **Spine Angle** | `mid_hip → mid_shoulder → v_ref` | Same vertical reference above mid-shoulder | Lean of the spine from vertical |

   For hip/shoulder/spine angles, a virtual reference point is used to measure against a horizontal or vertical baseline — no separate helper function needed.

5. **Drawing** — skeleton lines, landmark dots, and text labels are drawn onto the frame with OpenCV. Green text = ear-hip-wrist; cyan text = the three new angles.

6. Returns the annotated frame and a `metrics` dict:
   ```python
   {
     "wrist_x": int,
     "wrist_y": int,
     "ear_hip_wrist_deg": float,
     "hip_rotation_deg": float,
     "shoulder_tilt_deg": float,
     "spine_angle_deg": float,
   }
   ```
   Returns an empty dict `{}` if no pose was detected in the frame.

---

#### Swing Phase Detection — `detect_swing_phases(wrist_x_list, wrist_y_list, frame_numbers)`

Runs once after all frames are processed, on the collected wrist trajectory. Returns a dict mapping phase names to actual frame numbers.

**Signal processing steps:**

1. **Smooth** — applies a moving-average kernel (`window = max(3, n // 20)`) using `np.convolve` with `mode='same'` to suppress per-frame jitter.

2. **Impact** — `np.argmax(y_smooth)`. In image coordinates y increases downward, so the wrist is at its lowest physical position when `y` is at its maximum value.

3. **Top of backswing** — computed from `dx = np.diff(x_smooth)`. A sign change in `dx` means the wrist reversed horizontal direction. The last such reversal before the impact frame is the top:
   ```python
   sign_changes = np.where(np.diff(np.sign(dx)))[0] + 1
   top_idx = last sign_change before impact_idx
   ```
   The `+ 1` corrects for the off-by-one between the `diff` output and the original frame indices.

4. **Address end** — scans forward from frame 0 looking for the first frame where `|dx|` exceeds `2 × 40th-percentile speed` of the pre-top segment. The frame before that is the end of address.

5. The remaining phases (backswing, downswing, follow-through) are inferred from those three anchor points.

All list indices are mapped back to real frame numbers via the `frame_numbers` list before returning, correctly handling frames where pose detection failed (which are absent from the list).

---

#### JSON Export — `export_swing_data(output_path, video_path, fps, phases, frame_data)`

Builds and writes the final JSON file:

```json
{
  "video": "input/Reference_Swing_DTL.mp4",
  "fps": 30,
  "total_frames": 120,
  "phases": {
    "address":        { "start_frame": 0,  "end_frame": 12 },
    "backswing":      { "start_frame": 13, "end_frame": 44 },
    "top":            { "frame": 45 },
    "downswing":      { "start_frame": 46, "end_frame": 57 },
    "impact":         { "frame": 58 },
    "follow_through": { "start_frame": 59, "end_frame": 119 }
  },
  "metrics_at_top": {
    "hip_rotation_deg": 42.3,
    "shoulder_tilt_deg": 87.1,
    "spine_angle_deg": 32.5
  },
  "metrics_at_impact": {
    "hip_rotation_deg": 38.1,
    "wrist_speed_px_per_frame": 24.7
  },
  "frame_data": [
    { "frame": 0, "wrist_x": 610, "wrist_y": 830, "ear_hip_wrist_deg": 121.4, ... },
    ...
  ]
}
```

- `metrics_at_top` and `metrics_at_impact` are extracted from `frame_data` using a `frame_number → entry` lookup dict.
- `wrist_speed_px_per_frame` at impact is the Euclidean distance between the wrist position in the impact frame and the frame before it.

---

## Coordinate System

The on-screen axes drawn in the bottom-left corner reflect OpenCV's native coordinate system:

- **Origin (0, 0)** — top-left corner of the frame
- **X increases rightward**
- **Y increases downward**

This means "wrist at lowest point in the swing" = **maximum Y value**, which is why `np.argmax(y_smooth)` correctly identifies impact.

---

## Dependencies

| Package | Role |
|---|---|
| `opencv-python` | Video decode/encode, drawing, color conversion |
| `mediapipe` | Pose landmark detection (33-point body model) |
| `numpy` | Array math, smoothing, diff/sign-change detection |

---

## What's Next (Planned Phases)

| Phase | Description |
|---|---|
| Phase 3 | React frontend — chart frame_data[], display annotated video |
| Phase 4 | LLM coaching layer — feed swing_data.json to an LLM for natural-language feedback |
