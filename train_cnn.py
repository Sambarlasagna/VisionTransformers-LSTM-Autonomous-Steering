import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loader import UdacityDrivingDataset
from model_cnn import SteeringCNN

# Paths
DATASET_DIR = r"D:\Programming\Projects\VisionTransformers-LSTM-Autonomous-Steering\udacity_dataset\self_driving_car_dataset_make"
CSV_PATH = DATASET_DIR + r"\driving_log.csv"
IMG_DIR = DATASET_DIR + r"\IMG"

# Dataset
dataset = UdacityDrivingDataset(CSV_PATH, IMG_DIR)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SteeringCNN().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

best_loss = float("inf")
SAVE_PATH = "cnn_delta_model.pth"

# Training loop
for epoch in range(10):
    total_loss = 0.0

    for images, angles in loader:
        images = images.to(device)
        angles = angles.to(device)

        preds = model(images)
        loss = criterion(preds, angles)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.5f}")

    # Save the model if the loss is the best we've seen so far
    if total_loss < best_loss:
        best_loss = total_loss
        torch.save(model.state_dict(), SAVE_PATH)
