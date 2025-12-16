import os
import cv2
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class UdacityDrivingDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (str): Path to driving_log.csv
            img_dir (str): Path to IMG folder
            transform (callable, optional): Optional image transforms
        """
        self.data = pd.read_csv(csv_file, header=None)
        self.data.columns = [
            "center", "left", "right",
            "steering", "throttle", "brake", "speed"
        ]
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # ---- Image path handling (Kaggle-safe) ----
        img_name = os.path.basename(row["center"])
        img_path = os.path.join(self.img_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # BGR → RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize for ViT / CNN
        image = cv2.resize(image, (224, 224))

        # Normalize to [0,1]
        image = image.astype(np.float32) / 255.0

        # HWC → CHW
        image = torch.tensor(image).permute(2, 0, 1)

        steering = torch.tensor(row["steering"], dtype=torch.float32)

        return image, steering
