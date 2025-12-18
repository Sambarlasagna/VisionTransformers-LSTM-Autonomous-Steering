import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sequence_dataloader import UdacitySequenceDataset
from model_vit_lstm import SteeringViTLSTM

print("Starting ViT + LSTM training script...")

# Paths
DATASET_DIR = r"D:\Programming\Projects\VisionTransformers-LSTM-Autonomous-Steering\udacity_dataset\self_driving_car_dataset_make"
CSV_PATH = DATASET_DIR + r"\driving_log.csv"
IMG_DIR = DATASET_DIR + r"\IMG"

# Dataset
dataset = UdacitySequenceDataset(
    CSV_PATH,
    IMG_DIR,
    sequence_length=5
)

print("Dataset size:", len(dataset))

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0
)

print("Batches per epoch:", len(loader))

# Model
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = SteeringViTLSTM(sequence_length=5).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# <<< NEW: tracking best model
best_loss = float("inf")
SAVE_PATH = "vit_lstm_best.pth"

# Training
epochs = 3   # <<< YOU CAN CHANGE THIS SAFELY NOW

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1} started")
    total_loss = 0.0

    for batch_idx, (sequences, angles) in enumerate(loader):

        if batch_idx == 0:
            print("First batch loaded")
            print("Sequences shape:", sequences.shape)
            print("Angles shape:", angles.shape)

        sequences = sequences.to(device)
        angles = angles.to(device)

        preds = model(sequences)
        loss = criterion(preds, angles)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    epoch_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.5f}")

    # <<< NEW: save best model
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"✅ Best model saved (loss = {best_loss:.5f})")

print("\nTraining complete.")
print(f"Best model saved at: {SAVE_PATH}")
