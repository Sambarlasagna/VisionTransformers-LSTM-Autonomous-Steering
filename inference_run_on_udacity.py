import os
import cv2
import torch
import numpy as np
from collections import deque
from model_vit_lstm import SteeringViTLSTM

# ---------------- CONFIG ----------------
RUN_DIR = "run"
IMG_DIR = os.path.join(RUN_DIR, "IMG")
MODEL_PATH = "vit_lstm_best.pth"
SEQUENCE_LENGTH = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ----------------------------------------

# Load model
model = SteeringViTLSTM(sequence_length=SEQUENCE_LENGTH).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("Model loaded. Running inference on Udacity run...")

# Load image list
image_files = sorted([
    img for img in os.listdir(IMG_DIR)
    if img.endswith(".jpg")
])

frame_buffer = deque(maxlen=SEQUENCE_LENGTH)

def preprocess(frame):
    h, w, _ = frame.shape
    frame = frame[int(h*0.35):int(h*0.85), :]  # Udacity crop
    frame = cv2.resize(frame, (224, 224))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.float32) / 255.0
    frame = torch.tensor(frame).permute(2, 0, 1)
    return frame

def draw_steering(frame, angle):
    h, w, _ = frame.shape
    center = (w // 2, h - 40)
    length = 80

    visual_angle = angle * 5.0  # amplify for visibility
    angle_rad = -visual_angle * np.pi / 2

    end_x = int(center[0] + length * np.sin(angle_rad))
    end_y = int(center[1] - length * np.cos(angle_rad))

    cv2.arrowedLine(frame, center, (end_x, end_y), (0, 255, 0), 3)
    cv2.putText(frame, f"Steering: {angle:.4f}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)

# Run inference
for img_name in image_files:
    
    img_path = os.path.join(IMG_DIR, img_name)
    frame = cv2.imread(img_path)

    processed = preprocess(frame)
    frame_buffer.append(processed)

    if len(frame_buffer) == SEQUENCE_LENGTH:
        sequence = torch.stack(list(frame_buffer)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            steering = model(sequence).item()
    else:
        steering = 0.0
    
    print(f"{img_name} → {steering:.6f}")


    draw_steering(frame, steering)
    cv2.imshow("Udacity Run – ViT + LSTM Steering", frame)

    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
