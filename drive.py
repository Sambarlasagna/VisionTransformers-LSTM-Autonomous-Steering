import cv2
import torch
import numpy as np
from collections import deque

# -----------------------------
# CONFIG
VIDEO_INPUT = "./videos/udacity_forest_drive.mp4"          # input driving video
VIDEO_OUTPUT = "./videos/output_forest_autonomous.mp4"  # output video
MODEL_PATH = "models/nvidia_model.pth"  # trained PyTorch model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# LOAD MODEL
# -----------------------------
from model import NvidiaModel   # your PyTorch model definition

model = NvidiaModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# -----------------------------
# IMAGE PREPROCESSING
# (same as training)
# -----------------------------
def img_preprocess(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img / 255.0
    return img

# -----------------------------
# DRAW STEERING ARROW
# -----------------------------
def draw_steering_arrow(frame, steering):
    h, w, _ = frame.shape

    center_x = w // 2
    center_y = h - 40

    length = 120
    angle = steering * 45  # degrees

    rad = np.deg2rad(angle)

    end_x = int(center_x + length * np.sin(rad))
    end_y = int(center_y - length * np.cos(rad))

    cv2.arrowedLine(
        frame,
        (center_x, center_y),
        (end_x, end_y),
        (0, 0, 255),
        5,
        tipLength=0.3
    )

    return frame

# -----------------------------
# SMOOTHING (EMA)
# -----------------------------
steering_history = deque(maxlen=5)

def smooth_steering(value):
    steering_history.append(value)
    return np.mean(steering_history)

# -----------------------------
# VIDEO PIPELINE
# -----------------------------
cap = cv2.VideoCapture(VIDEO_INPUT)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, fps, (width, height))

print("[INFO] Running inference...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # preprocess
    img = img_preprocess(frame)
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # inference
    with torch.no_grad():
        steering = model(img).item()

    steering = np.clip(steering, -1.0, 1.0)
    steering = smooth_steering(steering)

    # draw
    frame = draw_steering_arrow(frame, steering)

    out.write(frame)

cap.release()
out.release()

print("[DONE] Output saved to:", VIDEO_OUTPUT)
