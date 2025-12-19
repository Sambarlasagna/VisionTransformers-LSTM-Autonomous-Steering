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
        self.steering_values = self.data["steering"].values.astype(np.float32)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
            row = self.data.iloc[idx]

            img_name = os.path.basename(row["center"])
            img_path = os.path.join(self.img_dir, img_name)

            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Image not found: {img_path}")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            image = image.astype(np.float32) / 255.0
            image = torch.tensor(image).permute(2, 0, 1)

            current_steering = self.steering_values[idx]
            if idx == 0:
                delta = 0.0
            else:
                delta = current_steering - self.steering_values[idx - 1]

            delta = torch.tensor(delta, dtype=torch.float32)
            return image, delta

