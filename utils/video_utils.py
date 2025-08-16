import cv2

def create_video_capture(file_path):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise IOError(f"Unable to open video: {file_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (frame_w, frame_h)

    return cap, fps, frame_size

def create_video_writer(output_path, fps, frame_size):
    return cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)
