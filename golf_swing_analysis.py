import cv2
import os
from utils.video_utils import create_video_capture, create_video_writer
from utils.swing_utils import process_frame

# Input and output paths
input_video = "input/Reference_Swing_DTL.mp4"
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Setup video capture and writer
cap, fps, frame_size = create_video_capture(input_video)
output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(input_video))[0] + "_out_landmarks.mp4")
writer = create_video_writer(output_path, fps, frame_size)

print("Processing frames, please wait...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    annotated_frame = process_frame(frame, frame_size)
    writer.write(annotated_frame)

print("Processing completed. Output saved at:", output_path)

cap.release()
writer.release()
cv2.destroyAllWindows()
