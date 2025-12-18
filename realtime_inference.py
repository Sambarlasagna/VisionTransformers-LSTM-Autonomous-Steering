import cv2
import torch
import numpy as np
from collections import deque
from model_vit_lstm import SteeringViTLSTM

# ---------------- CONFIG ----------------
MODEL_PATH = "vit_lstm_best.pth"
SEQUENCE_LENGTH = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VIDEO_PATH = "still_driving_video.mp4"   # 0 = webcam, or replace with video file path
# ----------------------------------------

# Load model
model = SteeringViTLSTM(sequence_length=SEQUENCE_LENGTH).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("Model loaded. Starting inference...")

# Frame buffer (rolling window)
frame_buffer = deque(maxlen=SEQUENCE_LENGTH)

# OpenCV video
cap = cv2.VideoCapture(VIDEO_PATH)

def preprocess(frame):
    # Crop top sky & bottom hood (Udacity-style)
    h, w, _ = frame.shape
    frame = frame[int(h*0.35):int(h*0.85), :]  # keep road region

    frame = cv2.resize(frame, (224, 224))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.float32) / 255.0
    frame = torch.tensor(frame).permute(2, 0, 1)
    return frame


def draw_steering(frame, angle):
    h, w, _ = frame.shape
    center = (w // 2, h - 50)
    length = 100

    visual_angle = angle * 5.0   # amplify for display
    angle_rad = -visual_angle * np.pi / 2

    end_x = int(center[0] + length * np.sin(angle_rad))
    end_y = int(center[1] - length * np.cos(angle_rad))

    cv2.arrowedLine(frame, center, (end_x, end_y), (0, 255, 0), 4)
    cv2.putText(frame, f"Steering: {angle:.4f}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    processed = preprocess(frame)
    frame_buffer.append(processed)

    if len(frame_buffer) == SEQUENCE_LENGTH:
        sequence = torch.stack(list(frame_buffer))
        sequence = sequence.unsqueeze(0).to(DEVICE)  # (1, T, 3, 224, 224)

        with torch.no_grad():
            steering = model(sequence).item()
    else:
        steering = 0.0

    draw_steering(frame, steering)
    cv2.imshow("ViT + LSTM Steering Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
