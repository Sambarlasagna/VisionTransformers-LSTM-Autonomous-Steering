import os
import cv2
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class UdacityDrivingDataset(Dataset):
    def __init__(self, csv_file, img_dir):
        self.img_dir = img_dir
        df = pd.read_csv(csv_file, header=None)
        df.columns = [
            "center", "left", "right",
            "steering", "throttle", "brake", "speed"
        ]

        STEERING_OFFSET = 0.2
        self.samples = []

        for _, row in df.iterrows():
            self.samples.append((row["center"], row["steering"]))
            self.samples.append((row["left"], row["steering"] + STEERING_OFFSET))
            self.samples.append((row["right"], row["steering"] - STEERING_OFFSET))

        self.steering_values = np.array(
            [s[1] for s in self.samples],
            dtype=np.float32
        )

    def __len__(self):
        return len(self.samples) - 1

    def __getitem__(self, idx):
        idx = idx + 1  # ensure idx-1 valid

        img_path, steering = self.samples[idx]
        prev_steering = self.steering_values[idx - 1]

        image = cv2.imread(os.path.join(self.img_dir, os.path.basename(img_path)))
        if image is None:
            raise FileNotFoundError(img_path)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0
        image = torch.tensor(image).permute(2, 0, 1)

        delta = steering - prev_steering
        return image, torch.tensor(delta, dtype=torch.float32)





