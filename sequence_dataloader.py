import os
import cv2
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class UdacitySequenceDataset(Dataset):
    def __init__(self, csv_file, img_dir, sequence_length=5):
        self.data = pd.read_csv(csv_file, header=None)
        self.data.columns = [
            "center", "left", "right",
            "steering", "throttle", "brake", "speed"
        ]
        self.img_dir = img_dir
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length + 1

    def _load_image(self, img_path):
        img_name = os.path.basename(img_path)
        full_path = os.path.join(self.img_dir, img_name)

        img = cv2.imread(full_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = torch.tensor(img).permute(2, 0, 1)

        return img

    def __getitem__(self, idx):
        images = []

        for i in range(idx, idx + self.sequence_length):
            img = self._load_image(self.data.iloc[i]["center"])
            images.append(img)

        images = torch.stack(images)  # (T, 3, 224, 224)

        steering = torch.tensor(
            self.data.iloc[idx + self.sequence_length - 1]["steering"],
            dtype=torch.float32
        )

        return images, steering
