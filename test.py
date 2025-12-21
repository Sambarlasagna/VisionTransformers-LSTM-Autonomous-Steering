import os
import cv2
import torch
import numpy as np
from model_cnn import SteeringCNN

# -------- CONFIG --------
IMG_PATH = r"D:\Programming\Projects\VisionTransformers-LSTM-Autonomous-Steering\RUN\IMG\right_2025_12_21_19_29_29_516.jpg" #change this
MODEL_PATH = "cnn_delta_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ------------------------

def preprocess(frame):
    h, w, _ = frame.shape
    frame = frame[int(h*0.35):int(h*0.85), :]
    frame = cv2.resize(frame, (224, 224))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.float32) / 255.0
    frame = torch.tensor(frame).permute(2, 0, 1)
    return frame.unsqueeze(0)  # (1, 3, 224, 224)

def main():
    # Load model
    model = SteeringCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Load image
    frame = cv2.imread(IMG_PATH)
    if frame is None:
        raise FileNotFoundError(f"Image not found: {IMG_PATH}")

    x = preprocess(frame).to(DEVICE)

    with torch.no_grad():
        delta = model(x).item()

    print("Predicted Δ steering:", delta)

if __name__ == "__main__":
    main()
